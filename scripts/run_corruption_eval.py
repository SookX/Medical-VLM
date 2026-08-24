from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.controller.dependency_audit import audit_accepted_chain
from cxreason.evaluation.metrics import run_record_from_state, summarize_run_records
from cxreason.evaluation.logging import append_jsonl, iter_attempt_records, write_json
from cxreason.evaluation.metadata import write_run_metadata
from cxreason.mock_cases.chexstruct import (
    CorruptingCheXStructGenerator,
    build_chexstruct_mock_case,
)
from cxreason.pipelines.tasks import build_task_controller
from cxreason.tasks.registry import get_task_spec, list_task_names


MODE_TO_STAGE = {
    "clean": None,
    "stage1": "stage1_criterion",
    "stage2": "stage2_anatomy",
    "stage3": "stage3_measurement",
    "stage4": "stage4_final",
}


def parse_tasks(values: list[str] | None) -> tuple[str, ...]:
    if not values or values == ["all"]:
        return list_task_names()
    return tuple(values)


def parse_modes(values: list[str] | None) -> tuple[str, ...]:
    if not values or values == ["all"]:
        return tuple(MODE_TO_STAGE)
    return tuple(values)


def build_generator(case: Any, spec: Any, mode: str) -> CorruptingCheXStructGenerator:
    return CorruptingCheXStructGenerator(
        case,
        spec=spec,
        corrupt_stage1_once=mode == "stage1",
        corrupt_stage2_once=mode == "stage2",
        corrupt_stage3_once=mode == "stage3",
        corrupt_stage4_once=mode == "stage4",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled corruption evaluation.")
    parser.add_argument("--tasks", nargs="+", default=["all"], choices=("all", *list_task_names()))
    parser.add_argument("--modes", nargs="+", default=["all"], choices=("all", *MODE_TO_STAGE))
    parser.add_argument("--source-dataset", default="nih_cxr14")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/corruption_eval")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    attempts_path = output_dir / "attempts.jsonl"
    runs_path = output_dir / "runs.jsonl"
    if attempts_path.exists():
        attempts_path.unlink()
    if runs_path.exists():
        runs_path.unlink()
    write_run_metadata(output_dir, config=vars(args))

    run_records: list[dict[str, Any]] = []

    for task in parse_tasks(args.tasks):
        spec = get_task_spec(task)
        controller = build_task_controller(task, max_attempts=args.max_attempts)
        for row_offset in range(args.count):
            row_index = args.row_index + row_offset
            case = build_chexstruct_mock_case(task, args.source_dataset, row_index)
            for mode in parse_modes(args.modes):
                corrupted_stage = MODE_TO_STAGE[mode]
                generator = build_generator(case, spec, mode)
                state = controller.run(case.case_id, case.task, generator)
                audit_result = audit_accepted_chain(state, spec)
                run_id = f"{task}:{case.case_id}:{row_index}:{mode}"
                verification_levels = spec.verification_levels

                append_jsonl(
                    attempts_path,
                    iter_attempt_records(
                        state,
                        run_id=run_id,
                        mode=mode,
                        verification_levels=verification_levels,
                        corrupted_stage=corrupted_stage,
                        extra_fields={"system": "local_repair"},
                    ),
                )

                run_record = run_record_from_state(
                    state,
                    system="local_repair",
                    run_id=run_id,
                    mode=mode,
                    stage_names=spec.stage_names,
                    corrupted_stage=corrupted_stage,
                    verification_level=(
                        verification_levels.get(corrupted_stage) if corrupted_stage else None
                    ),
                    dependency_audit_passed=audit_result.passed,
                    dependency_audit_reason=audit_result.reason,
                    extra={"row_index": row_index},
                )
                run_records.append(run_record)
                append_jsonl(runs_path, [run_record])

    summary = summarize_run_records(
        run_records,
        group_fields=("task", "mode", "corrupted_stage", "verification_level"),
    )
    write_json(output_dir / "summary.json", summary)

    overall = summary["overall"]
    print(f"runs: {overall['cases']}")
    print(f"detected: {overall['detected']}")
    print(f"completed: {overall['completed']}")
    print(f"false_repairs: {overall['false_repairs']}")
    print(f"attempts_jsonl: {attempts_path}")
    print(f"runs_jsonl: {runs_path}")
    print(f"summary_json: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

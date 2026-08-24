from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.controller.dependency_audit import audit_accepted_chain
from cxreason.evaluation.logging import append_jsonl, iter_attempt_records, write_json
from cxreason.evaluation.metadata import write_run_metadata
from cxreason.evaluation.metrics import run_record_from_state, summarize_run_records
from cxreason.mock_cases.chexstruct import (
    CorruptingCheXStructGenerator,
    build_chexstruct_mock_case,
)
from cxreason.pipelines.tasks import build_full_restart_task_controller, build_task_controller
from cxreason.tasks.registry import get_task_spec, list_task_names
from scripts.run_corruption_eval import MODE_TO_STAGE, parse_modes, parse_tasks


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
    parser = argparse.ArgumentParser(description="Compare local repair with full-chain restart.")
    parser.add_argument("--tasks", nargs="+", default=["all"], choices=("all", *list_task_names()))
    parser.add_argument("--modes", nargs="+", default=["stage3", "stage4"], choices=("all", *MODE_TO_STAGE))
    parser.add_argument("--source-dataset", default="nih_cxr14")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/local_vs_full_restart")
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
        controllers = {
            "local_repair": build_task_controller(task, max_attempts=args.max_attempts),
            "full_restart": build_full_restart_task_controller(
                task,
                max_trajectories=args.max_attempts,
            ),
        }
        verification_levels = spec.verification_levels

        for row_offset in range(args.count):
            row_index = args.row_index + row_offset
            case = build_chexstruct_mock_case(task, args.source_dataset, row_index)

            for mode in parse_modes(args.modes):
                corrupted_stage = MODE_TO_STAGE[mode]
                for system, controller in controllers.items():
                    generator = build_generator(case, spec, mode)
                    state = controller.run(case.case_id, case.task, generator)
                    audit_result = audit_accepted_chain(state, spec)
                    run_id = f"{system}:{task}:{case.case_id}:{row_index}:{mode}"

                    append_jsonl(
                        attempts_path,
                        iter_attempt_records(
                            state,
                            run_id=run_id,
                            mode=mode,
                            verification_levels=verification_levels,
                            corrupted_stage=corrupted_stage,
                            extra_fields={"system": system},
                        ),
                    )
                    run_record = run_record_from_state(
                        state,
                        system=system,
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

    summary = summarize_run_records(run_records)
    system_summary = summarize_run_records(
        run_records,
        group_fields=("system", "mode", "verification_level"),
    )
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "system_summary.json", system_summary)

    print(f"runs: {summary['overall']['cases']}")
    for row in system_summary["by_group"]:
        print(
            f"{row['system']} {row['mode']} {row['verification_level']}: "
            f"completion={row['completion_rate']:.3f} "
            f"mean_calls={row['mean_model_calls']:.2f} "
            f"depth={row['mean_reasoning_depth']:.2f}"
        )
    print(f"attempts_jsonl: {attempts_path}")
    print(f"runs_jsonl: {runs_path}")
    print(f"summary_json: {output_dir / 'summary.json'}")
    print(f"system_summary_json: {output_dir / 'system_summary.json'}")


if __name__ == "__main__":
    main()

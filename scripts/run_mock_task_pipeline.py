from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.mock_cases.chexstruct import (
    CorruptingCheXStructGenerator,
    build_chexstruct_mock_case,
)
from cxreason.pipelines.tasks import build_task_controller
from cxreason.tasks.registry import get_task_spec, list_task_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CheXStruct-backed mock task pipeline.")
    parser.add_argument("--task", choices=list_task_names(), default="cardiomegaly")
    parser.add_argument("--source-dataset", default="nih_cxr14")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--no-corrupt-stage3", action="store_true")
    parser.add_argument("--corrupt-stage1", action="store_true")
    parser.add_argument("--corrupt-stage2", action="store_true")
    parser.add_argument("--corrupt-stage4", action="store_true")
    args = parser.parse_args()

    spec = get_task_spec(args.task)
    controller = build_task_controller(args.task, max_attempts=args.max_attempts)
    passed = 0
    total_model_calls = 0
    total_verifier_calls = 0

    for offset in range(args.count):
        case = build_chexstruct_mock_case(args.task, args.source_dataset, args.row_index + offset)
        generator = CorruptingCheXStructGenerator(
            case,
            spec=spec,
            corrupt_stage1_once=args.corrupt_stage1,
            corrupt_stage2_once=args.corrupt_stage2,
            corrupt_stage3_once=not args.no_corrupt_stage3,
            corrupt_stage4_once=args.corrupt_stage4,
        )
        state = controller.run(case.case_id, case.task, generator)
        passed += int(state.passed)
        total_model_calls += state.model_calls
        total_verifier_calls += state.verifier_calls

        if args.count == 1:
            print(f"case_id: {state.case_id}")
            print(f"task: {state.task}")
            print(f"passed: {state.passed}")
            print(f"failed_stage: {state.failed_stage}")
            print(f"model_calls: {state.model_calls}")
            print(f"verifier_calls: {state.verifier_calls}")
            for stage in spec.stage_names:
                if stage not in state.stage_outputs:
                    continue
                print(f"{stage}: attempts={state.attempts.get(stage, 0)}")
                for attempt_idx, gate_result in enumerate(state.gate_results[stage], start=1):
                    print(
                        f"  attempt {attempt_idx}: "
                        f"passed={gate_result.passed} reason={gate_result.reason}"
                    )
                if stage in state.accepted_outputs:
                    print(f"  accepted={state.accepted_outputs[stage]}")

    if args.count > 1:
        print(f"task: {args.task}")
        print(f"cases: {args.count}")
        print(f"passed: {passed}")
        print(f"completion: {passed / args.count:.3f}")
        print(f"model_calls: {total_model_calls}")
        print(f"verifier_calls: {total_verifier_calls}")


if __name__ == "__main__":
    main()

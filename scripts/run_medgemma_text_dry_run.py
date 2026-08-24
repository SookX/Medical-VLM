from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.generators.medgemma_task import MedGemmaTaskGenerator
from cxreason.mock_cases.chexstruct import build_chexstruct_mock_case
from cxreason.modeling.medgemma import MedGemmaAdapter, MedGemmaConfig
from cxreason.pipelines.tasks import build_task_controller
from cxreason.prompting.tasks import build_task_stage_prompt
from cxreason.tasks.registry import get_task_spec, list_task_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or render a MedGemma text-only dry run.")
    parser.add_argument("--task", choices=list_task_names(), default="cardiomegaly")
    parser.add_argument("--source-dataset", default="nih_cxr14")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--model-id", default="google/medgemma-4b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--call-model", action="store_true")
    args = parser.parse_args()

    spec = get_task_spec(args.task)
    case = build_chexstruct_mock_case(args.task, args.source_dataset, args.row_index)

    if not args.call_model:
        for stage in spec.stage_names:
            print(f"===== {stage} =====")
            print(
                build_task_stage_prompt(
                    args.task,
                    stage,
                    accepted_outputs={},
                    development_context=case.stage_outputs.get(stage),
                )
            )
        return

    model = MedGemmaAdapter(
        MedGemmaConfig(
            model_id=args.model_id,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
        )
    )
    generator = MedGemmaTaskGenerator(
        args.task,
        model,
        image_path=None,
        development_context=case.stage_outputs,
    )
    state = build_task_controller(args.task, args.max_attempts).run(
        case.case_id,
        case.task,
        generator,
    )

    print(f"case_id: {state.case_id}")
    print(f"task: {state.task}")
    print(f"passed: {state.passed}")
    print(f"failed_stage: {state.failed_stage}")
    for stage in spec.stage_names:
        if stage in state.stage_outputs:
            print(f"{stage}: attempts={state.attempts[stage]}")
            print(state.stage_outputs[stage][-1])


if __name__ == "__main__":
    main()

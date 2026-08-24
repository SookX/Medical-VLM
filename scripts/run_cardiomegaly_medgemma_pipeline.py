from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.generators.medgemma_cardiomegaly import MedGemmaCardiomegalyGenerator
from cxreason.mock_cases.cardiomegaly import CARDIOMEGALY_STAGE_NAMES
from cxreason.modeling.medgemma import MedGemmaAdapter, MedGemmaConfig
from cxreason.pipelines.cardiomegaly import build_cardiomegaly_controller


def printable_output(output: Any, show_raw: bool) -> Any:
    if show_raw or not isinstance(output, dict):
        return output
    return {key: value for key, value in output.items() if key != "_raw_text"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedGemma cardiomegaly Path-1 pipeline.")
    parser.add_argument("--image", required=True, help="Path to a chest X-ray image.")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--model-id", default="google/medgemma-4b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--show-raw", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image)
    config = MedGemmaConfig(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
    )
    model = MedGemmaAdapter(config)
    generator = MedGemmaCardiomegalyGenerator(model, image_path)
    controller = build_cardiomegaly_controller(args.max_attempts)

    case_id = args.case_id or image_path.stem
    state = controller.run(case_id=case_id, task="cardiomegaly", generator=generator)

    print(f"case_id: {state.case_id}")
    print(f"passed: {state.passed}")
    print(f"failed_stage: {state.failed_stage}")
    print(f"model_calls: {state.model_calls}")
    print(f"verifier_calls: {state.verifier_calls}")

    for stage in CARDIOMEGALY_STAGE_NAMES:
        if stage not in state.stage_outputs:
            continue
        print(f"{stage}: attempts={state.attempts.get(stage, 0)}")
        for attempt_idx, output in enumerate(state.stage_outputs[stage], start=1):
            gate_result = state.gate_results[stage][attempt_idx - 1]
            print(
                f"  attempt {attempt_idx}: "
                f"passed={gate_result.passed} reason={gate_result.reason}"
            )
            print(f"  output={printable_output(output, args.show_raw)}")


if __name__ == "__main__":
    main()

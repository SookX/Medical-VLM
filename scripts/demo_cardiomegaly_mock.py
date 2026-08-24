from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.gates.cardiomegaly import (
    CTRArithmeticGate,
    CardiomegalyCriterionGate,
    CardiomegalyRuleGate,
)
from cxreason.mock_cases.cardiomegaly import build_cardiomegaly_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo cardiomegaly mock Path-1 gates.")
    parser.add_argument("--source-dataset", default="nih_cxr14")
    parser.add_argument("--row-index", type=int, default=0)
    args = parser.parse_args()

    case = build_cardiomegaly_case(args.source_dataset, args.row_index)
    answers = case.stage_answers

    criterion_result = CardiomegalyCriterionGate().verify(answers["stage1_criterion"], {})
    measurement_result = CTRArithmeticGate().verify(answers["stage3_measurement"], answers)
    final_result = CardiomegalyRuleGate().verify(
        answers["stage4_final"],
        {"measurement": answers["stage3_measurement"]},
    )

    print(f"case_id: {case.case_id}")
    print(f"task: {case.task}")
    print(f"criterion: {answers['stage1_criterion']} -> {criterion_result.passed}")
    print(f"anatomy: {answers['stage2_anatomy']}")
    print(f"measurement: {answers['stage3_measurement']} -> {measurement_result.passed}")
    print(f"final: {answers['stage4_final']} -> {final_result.passed}")


if __name__ == "__main__":
    main()

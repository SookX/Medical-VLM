from __future__ import annotations

import unittest

from cxreason.mock_cases.cardiomegaly import (
    CorruptingCardiomegalyGenerator,
    build_cardiomegaly_case,
)
from cxreason.gates.cardiomegaly import CardiomegalyRuleGate
from cxreason.pipelines.cardiomegaly import build_cardiomegaly_controller


class CardiomegalyMockPipelineTest(unittest.TestCase):
    def test_stage3_repair_preserves_accepted_upstream_stages(self) -> None:
        case = build_cardiomegaly_case(row_index=1)
        generator = CorruptingCardiomegalyGenerator(case, corrupt_stage3_once=True)

        state = build_cardiomegaly_controller().run(case.case_id, case.task, generator)

        self.assertTrue(state.passed)
        self.assertEqual(state.attempts["stage1_criterion"], 1)
        self.assertEqual(state.attempts["stage2_anatomy"], 1)
        self.assertEqual(state.attempts["stage3_measurement"], 2)
        self.assertEqual(state.attempts["stage4_final"], 1)
        self.assertFalse(state.gate_results["stage3_measurement"][0].passed)
        self.assertTrue(state.gate_results["stage3_measurement"][1].passed)

    def test_retry_budget_exhaustion_fails_current_stage(self) -> None:
        case = build_cardiomegaly_case(row_index=1)
        generator = CorruptingCardiomegalyGenerator(case, corrupt_stage3_once=True)

        state = build_cardiomegaly_controller(max_attempts=1).run(case.case_id, case.task, generator)

        self.assertFalse(state.passed)
        self.assertEqual(state.failed_stage, "stage3_measurement")
        self.assertNotIn("stage4_final", state.attempts)

    def test_final_gate_uses_trusted_viewposition_threshold(self) -> None:
        gate = CardiomegalyRuleGate()

        result = gate.verify(
            {"cardiomegaly": True, "viewposition": "PA", "threshold": 0.1},
            {
                "stage2_anatomy": {"viewposition": "AP"},
                "stage3_measurement": {"ctr": 0.51},
            },
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.metadata["threshold"], 0.545)


if __name__ == "__main__":
    unittest.main()

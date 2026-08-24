from __future__ import annotations

import unittest

from cxreason.gates.oracle import MockAnswerScorer, OracleGate


class OracleGateTest(unittest.TestCase):
    def test_oracle_gate_returns_only_pass_repair_signal(self) -> None:
        scorer = MockAnswerScorer({"stage1_criterion": {"criterion": "CTR"}})
        gate = OracleGate("stage1_criterion", scorer)

        result = gate.verify({"criterion": "wrong"}, {})

        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "oracle_stage_failed")
        self.assertNotIn("answer", result.metadata)

    def test_mock_oracle_accepts_matching_answer(self) -> None:
        scorer = MockAnswerScorer({"stage1_criterion": {"criterion": "CTR"}})
        gate = OracleGate("stage1_criterion", scorer)

        result = gate.verify({"criterion": "CTR"}, {})

        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()

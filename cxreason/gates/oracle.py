from __future__ import annotations

import math
from typing import Any, Protocol

from cxreason.gates.base import GateResult, StageGate


class BenchmarkScorer(Protocol):
    def is_correct(self, *, stage: str, output: Any, context: dict[str, Any]) -> bool:
        ...


class OracleGate(StageGate):
    """Oracle PASS/REPAIR gate that never exposes the hidden answer."""

    def __init__(self, stage: str, scorer: BenchmarkScorer) -> None:
        self.stage = stage
        self.scorer = scorer

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        passed = self.scorer.is_correct(
            stage=self.stage,
            output=stage_output,
            context=context,
        )
        return GateResult(
            passed=passed,
            reason=None if passed else "oracle_stage_failed",
            metadata={"verification_level": "oracle"},
        )


class MockAnswerScorer:
    """Mock oracle scorer for development with hidden structured answers."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers

    def is_correct(self, *, stage: str, output: Any, context: dict[str, Any]) -> bool:
        return _answers_equal(output, self.answers[stage])


def _answers_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_answers_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_answers_equal(a, b) for a, b in zip(left, right))
    return left == right

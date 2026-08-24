from __future__ import annotations

import math
from typing import Any, Iterable

from cxreason.gates.base import GateResult, StageGate


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


class CompositeGate(StageGate):
    """Run a sequence of gates and fail on the first failed check."""

    def __init__(self, gates: Iterable[StageGate]) -> None:
        self.gates = list(gates)

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        metadata: dict[str, Any] = {"subresults": []}
        max_score: float | None = None
        for gate in self.gates:
            result = gate.verify(stage_output, context)
            metadata["subresults"].append(
                {
                    "gate": gate.__class__.__name__,
                    "passed": result.passed,
                    "reason": result.reason,
                    "score": result.score,
                    "metadata": result.metadata,
                }
            )
            if result.score is not None:
                max_score = result.score if max_score is None else max(max_score, result.score)
            if not result.passed:
                return GateResult(
                    passed=False,
                    score=result.score,
                    reason=result.reason,
                    metadata=metadata,
                )
        return GateResult(passed=True, score=max_score, metadata=metadata)


class CriterionTermsGate(StageGate):
    """Accept criterion text containing one of the task-specific terms."""

    def __init__(self, terms: Iterable[str]) -> None:
        self.terms = tuple(term.lower() for term in terms)

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if isinstance(stage_output, dict):
            text = str(stage_output.get("criterion", stage_output)).lower()
        else:
            text = str(stage_output).lower()
        passed = any(term in text for term in self.terms)
        return GateResult(
            passed=passed,
            reason=None if passed else "criterion_term_not_found",
            metadata={"accepted_terms": self.terms},
        )


class RequiredFieldsGate(StageGate):
    """Check that a structured stage output contains required fields."""

    def __init__(
        self,
        required_fields: Iterable[str],
        numeric_fields: Iterable[str] = (),
        stage_label: str = "stage",
    ) -> None:
        self.required_fields = tuple(required_fields)
        self.numeric_fields = tuple(numeric_fields)
        self.stage_label = stage_label

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if not isinstance(stage_output, dict):
            return GateResult(passed=False, reason=f"{self.stage_label}_not_structured")

        missing = [
            field
            for field in self.required_fields
            if field not in stage_output or _missing(stage_output[field])
        ]
        if missing:
            return GateResult(
                passed=False,
                reason=f"{self.stage_label}_missing_required_fields",
                metadata={"missing": missing},
            )

        not_numeric = []
        for field in self.numeric_fields:
            try:
                float(stage_output[field])
            except (KeyError, TypeError, ValueError):
                not_numeric.append(field)

        if not_numeric:
            return GateResult(
                passed=False,
                reason=f"{self.stage_label}_non_numeric_fields",
                metadata={"fields": not_numeric},
            )

        return GateResult(passed=True)


class RatioConsistencyGate(StageGate):
    """Check target approximately equals numerator divided by denominator."""

    def __init__(
        self,
        target: str,
        numerator: str,
        denominator: str,
        tolerance: float = 0.01,
    ) -> None:
        self.target = target
        self.numerator = numerator
        self.denominator = denominator
        self.tolerance = tolerance

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if not isinstance(stage_output, dict):
            return GateResult(passed=False, reason="ratio_stage_not_structured")
        try:
            target = float(stage_output[self.target])
            numerator = float(stage_output[self.numerator])
            denominator = float(stage_output[self.denominator])
        except (KeyError, TypeError, ValueError) as exc:
            return GateResult(
                passed=False,
                reason="missing_or_invalid_ratio_fields",
                metadata={"error": str(exc)},
            )
        if denominator == 0:
            return GateResult(passed=False, reason="zero_ratio_denominator")

        expected = numerator / denominator
        error = abs(target - expected)
        passed = error <= self.tolerance
        return GateResult(
            passed=passed,
            score=error,
            reason=None if passed else "ratio_arithmetic_mismatch",
            metadata={
                "target": self.target,
                "expected": expected,
                "reported": target,
                "tolerance": self.tolerance,
            },
        )


class MinMaxRatioConsistencyGate(StageGate):
    """Check target equals min(a, b) divided by max(a, b)."""

    def __init__(
        self,
        target: str,
        value_a: str,
        value_b: str,
        tolerance: float = 0.01,
    ) -> None:
        self.target = target
        self.value_a = value_a
        self.value_b = value_b
        self.tolerance = tolerance

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if not isinstance(stage_output, dict):
            return GateResult(passed=False, reason="minmax_ratio_stage_not_structured")
        try:
            target = float(stage_output[self.target])
            value_a = float(stage_output[self.value_a])
            value_b = float(stage_output[self.value_b])
        except (KeyError, TypeError, ValueError) as exc:
            return GateResult(
                passed=False,
                reason="missing_or_invalid_minmax_ratio_fields",
                metadata={"error": str(exc)},
            )
        denominator = max(value_a, value_b)
        if denominator == 0:
            return GateResult(passed=False, reason="zero_minmax_ratio_denominator")

        expected = min(value_a, value_b) / denominator
        error = abs(target - expected)
        passed = error <= self.tolerance
        return GateResult(
            passed=passed,
            score=error,
            reason=None if passed else "minmax_ratio_arithmetic_mismatch",
            metadata={
                "target": self.target,
                "expected": expected,
                "reported": target,
                "tolerance": self.tolerance,
            },
        )


class BinaryDecisionFieldGate(StageGate):
    """Structural check for a final binary decision field."""

    def __init__(self, field: str) -> None:
        self.field = field

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if not isinstance(stage_output, dict):
            return GateResult(passed=False, reason="final_decision_not_structured")
        if self.field not in stage_output or _missing(stage_output[self.field]):
            return GateResult(
                passed=False,
                reason="missing_final_decision",
                metadata={"field": self.field},
            )
        if not isinstance(stage_output[self.field], bool):
            return GateResult(
                passed=False,
                reason="final_decision_not_boolean",
                metadata={"field": self.field, "value": stage_output[self.field]},
            )
        return GateResult(
            passed=True,
            metadata={"verification_level": "structural_final_decision"},
        )


class NumericRangeGate(StageGate):
    """Check numeric fields fall within broad plausible bounds."""

    def __init__(
        self,
        bounds: dict[str, tuple[float | None, float | None]],
        stage_label: str = "stage",
    ) -> None:
        self.bounds = bounds
        self.stage_label = stage_label

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if not isinstance(stage_output, dict):
            return GateResult(passed=False, reason=f"{self.stage_label}_not_structured")

        violations: dict[str, dict[str, float | None]] = {}
        for field, (low, high) in self.bounds.items():
            try:
                value = float(stage_output[field])
            except (KeyError, TypeError, ValueError):
                violations[field] = {"value": None, "low": low, "high": high}
                continue
            if (low is not None and value < low) or (high is not None and value > high):
                violations[field] = {"value": value, "low": low, "high": high}

        if violations:
            return GateResult(
                passed=False,
                reason=f"{self.stage_label}_numeric_range_violation",
                metadata={"violations": violations},
            )
        return GateResult(passed=True)


class BinaryFieldsGate(StageGate):
    """Check fields are numeric binary values."""

    def __init__(self, fields: Iterable[str], stage_label: str = "stage") -> None:
        self.fields = tuple(fields)
        self.stage_label = stage_label

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if not isinstance(stage_output, dict):
            return GateResult(passed=False, reason=f"{self.stage_label}_not_structured")

        violations = {}
        for field in self.fields:
            try:
                value = int(stage_output[field])
            except (KeyError, TypeError, ValueError):
                violations[field] = stage_output.get(field) if isinstance(stage_output, dict) else None
                continue
            if value not in (0, 1):
                violations[field] = value

        if violations:
            return GateResult(
                passed=False,
                reason=f"{self.stage_label}_binary_field_violation",
                metadata={"violations": violations},
            )
        return GateResult(passed=True)


class ChoiceFieldGate(StageGate):
    """Check a string field belongs to an accepted set."""

    def __init__(self, field: str, choices: Iterable[str], stage_label: str = "stage") -> None:
        self.field = field
        self.choices = tuple(choice.lower() for choice in choices)
        self.stage_label = stage_label

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if not isinstance(stage_output, dict):
            return GateResult(passed=False, reason=f"{self.stage_label}_not_structured")
        value = str(stage_output.get(self.field, "")).strip().lower()
        if value not in self.choices:
            return GateResult(
                passed=False,
                reason=f"{self.stage_label}_choice_field_violation",
                metadata={"field": self.field, "value": value, "choices": self.choices},
            )
        return GateResult(passed=True)

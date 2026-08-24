from __future__ import annotations

from typing import Any, Iterable

from cxreason.gates.base import GateResult, StageGate


def _measurement(context: dict[str, Any]) -> dict[str, Any] | None:
    measurement = context.get("stage3_measurement") or context.get("measurement")
    return measurement if isinstance(measurement, dict) else None


def _anatomy(context: dict[str, Any]) -> dict[str, Any]:
    anatomy = context.get("stage2_anatomy") or {}
    return anatomy if isinstance(anatomy, dict) else {}


def _decision(stage_output: Any, field: str) -> tuple[bool | None, GateResult | None]:
    if not isinstance(stage_output, dict):
        return None, GateResult(passed=False, reason="final_decision_not_structured")
    if field not in stage_output:
        return None, GateResult(
            passed=False,
            reason="missing_final_decision",
            metadata={"field": field},
        )
    if not isinstance(stage_output[field], bool):
        return None, GateResult(
            passed=False,
            reason="final_decision_not_boolean",
            metadata={"field": field, "value": stage_output[field]},
        )
    return stage_output[field], None


class MeasurementThresholdRuleGate(StageGate):
    """Verify a final boolean decision from an accepted numeric measurement."""

    def __init__(
        self,
        final_field: str,
        measurement_field: str,
        threshold: float,
        operator: str,
        viewposition_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.final_field = final_field
        self.measurement_field = measurement_field
        self.threshold = threshold
        self.operator = operator
        self.viewposition_thresholds = viewposition_thresholds or {}

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        decision, error = _decision(stage_output, self.final_field)
        if error is not None:
            return error

        measurement = _measurement(context)
        if measurement is None:
            return GateResult(passed=False, reason="missing_accepted_measurement")
        try:
            value = float(measurement[self.measurement_field])
        except (KeyError, TypeError, ValueError) as exc:
            return GateResult(
                passed=False,
                reason="invalid_accepted_measurement",
                metadata={"field": self.measurement_field, "error": str(exc)},
            )

        viewposition = str(_anatomy(context).get("viewposition", "")).upper()
        threshold = self.viewposition_thresholds.get(viewposition, self.threshold)
        expected = self._compare(value, threshold)
        passed = decision == expected
        return GateResult(
            passed=passed,
            reason=None if passed else "decision_does_not_follow_measurement_rule",
            metadata={
                "field": self.measurement_field,
                "value": value,
                "operator": self.operator,
                "threshold": threshold,
                "expected_decision": expected,
            },
        )

    def _compare(self, value: float, threshold: float) -> bool:
        if self.operator == ">":
            return value > threshold
        if self.operator == ">=":
            return value >= threshold
        if self.operator == "<":
            return value < threshold
        if self.operator == "<=":
            return value <= threshold
        raise ValueError(f"Unsupported threshold operator: {self.operator}")


class MeasurementRangeRuleGate(StageGate):
    """Verify final decision where positive means value is outside a normal range."""

    def __init__(
        self,
        final_field: str,
        measurement_field: str,
        low: float,
        high: float,
    ) -> None:
        self.final_field = final_field
        self.measurement_field = measurement_field
        self.low = low
        self.high = high

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        decision, error = _decision(stage_output, self.final_field)
        if error is not None:
            return error
        measurement = _measurement(context)
        if measurement is None:
            return GateResult(passed=False, reason="missing_accepted_measurement")
        try:
            value = float(measurement[self.measurement_field])
        except (KeyError, TypeError, ValueError) as exc:
            return GateResult(
                passed=False,
                reason="invalid_accepted_measurement",
                metadata={"field": self.measurement_field, "error": str(exc)},
            )
        expected = value < self.low or value > self.high
        passed = decision == expected
        return GateResult(
            passed=passed,
            reason=None if passed else "decision_does_not_follow_range_rule",
            metadata={
                "field": self.measurement_field,
                "value": value,
                "normal_low": self.low,
                "normal_high": self.high,
                "expected_decision": expected,
            },
        )


class AnyMeasurementThresholdRuleGate(StageGate):
    """Verify final decision from whether any accepted measurement crosses threshold."""

    def __init__(
        self,
        final_field: str,
        measurement_fields: Iterable[str],
        threshold: float,
        operator: str,
    ) -> None:
        self.final_field = final_field
        self.measurement_fields = tuple(measurement_fields)
        self.threshold = threshold
        self.operator = operator

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        decision, error = _decision(stage_output, self.final_field)
        if error is not None:
            return error
        measurement = _measurement(context)
        if measurement is None:
            return GateResult(passed=False, reason="missing_accepted_measurement")

        values: dict[str, float] = {}
        try:
            for field in self.measurement_fields:
                values[field] = float(measurement[field])
        except (KeyError, TypeError, ValueError) as exc:
            return GateResult(
                passed=False,
                reason="invalid_accepted_measurement",
                metadata={"error": str(exc)},
            )

        expected = any(self._compare(value) for value in values.values())
        passed = decision == expected
        return GateResult(
            passed=passed,
            reason=None if passed else "decision_does_not_follow_any_threshold_rule",
            metadata={
                "values": values,
                "operator": self.operator,
                "threshold": self.threshold,
                "expected_decision": expected,
            },
        )

    def _compare(self, value: float) -> bool:
        if self.operator == ">":
            return value > self.threshold
        if self.operator == ">=":
            return value >= self.threshold
        if self.operator == "<":
            return value < self.threshold
        if self.operator == "<=":
            return value <= self.threshold
        raise ValueError(f"Unsupported threshold operator: {self.operator}")


class AllBinaryMeasurementsRuleGate(StageGate):
    """Verify final decision equals all accepted binary measurement fields."""

    def __init__(self, final_field: str, measurement_fields: Iterable[str]) -> None:
        self.final_field = final_field
        self.measurement_fields = tuple(measurement_fields)

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        decision, error = _decision(stage_output, self.final_field)
        if error is not None:
            return error
        measurement = _measurement(context)
        if measurement is None:
            return GateResult(passed=False, reason="missing_accepted_measurement")

        values: dict[str, bool] = {}
        try:
            for field in self.measurement_fields:
                values[field] = bool(int(measurement[field]))
        except (KeyError, TypeError, ValueError) as exc:
            return GateResult(
                passed=False,
                reason="invalid_accepted_binary_measurement",
                metadata={"error": str(exc)},
            )

        expected = all(values.values())
        passed = decision == expected
        return GateResult(
            passed=passed,
            reason=None if passed else "decision_does_not_follow_all_binary_rule",
            metadata={"values": values, "expected_decision": expected},
        )


class DirectionNotFlatRuleGate(StageGate):
    """Verify final decision where deviation means direction is not flat."""

    def __init__(self, final_field: str, direction_field: str = "direction") -> None:
        self.final_field = final_field
        self.direction_field = direction_field

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        decision, error = _decision(stage_output, self.final_field)
        if error is not None:
            return error
        measurement = _measurement(context)
        if measurement is None:
            return GateResult(passed=False, reason="missing_accepted_measurement")
        direction = str(measurement.get(self.direction_field, "")).strip().lower()
        if not direction:
            return GateResult(
                passed=False,
                reason="missing_direction_measurement",
                metadata={"field": self.direction_field},
            )
        expected = direction != "flat"
        passed = decision == expected
        return GateResult(
            passed=passed,
            reason=None if passed else "decision_does_not_follow_direction_rule",
            metadata={"direction": direction, "expected_decision": expected},
        )

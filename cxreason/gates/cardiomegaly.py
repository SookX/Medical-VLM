from __future__ import annotations

from typing import Any

from cxreason.gates.base import GateResult, StageGate


class CardiomegalyCriterionGate(StageGate):
    """Accept only cardiothoracic-ratio criterion for cardiomegaly."""

    accepted_terms = ("cardiothoracic ratio", "ctr")

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if isinstance(stage_output, dict):
            text = str(stage_output.get("criterion", stage_output)).lower()
        else:
            text = str(stage_output).lower()
        passed = any(term in text for term in self.accepted_terms)
        return GateResult(passed=passed, reason=None if passed else "criterion_not_ctr")


class CardiomegalyAnatomyGate(StageGate):
    """Check that the anatomy stage provides heart and lung width evidence."""

    required_fields = ("heart_xmin", "heart_xmax", "lung_xmin", "lung_xmax")

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if not isinstance(stage_output, dict):
            return GateResult(passed=False, reason="anatomy_not_structured")
        missing = [field for field in self.required_fields if field not in stage_output]
        if missing:
            return GateResult(
                passed=False,
                reason="missing_cardiomegaly_anatomy",
                metadata={"missing": missing},
            )
        not_numeric = []
        for field in self.required_fields:
            try:
                float(stage_output[field])
            except (TypeError, ValueError):
                not_numeric.append(field)
        if not_numeric:
            return GateResult(
                passed=False,
                reason="non_numeric_cardiomegaly_anatomy",
                metadata={"fields": not_numeric},
            )
        return GateResult(passed=True)


class CTRArithmeticGate(StageGate):
    """Check reported CTR against reported heart and lung widths."""

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if not isinstance(stage_output, dict):
            return GateResult(passed=False, reason="measurement_not_structured")
        try:
            heart_width = float(stage_output["heart_width"])
            lung_width = float(stage_output["lung_width"])
            reported_ctr = float(stage_output["ctr"])
        except (KeyError, TypeError, ValueError) as exc:
            return GateResult(
                passed=False,
                reason="missing_or_invalid_ctr_measurement",
                metadata={"error": str(exc)},
            )
        if lung_width == 0:
            return GateResult(passed=False, reason="zero_lung_width")
        expected_ctr = heart_width / lung_width
        error = abs(reported_ctr - expected_ctr)
        passed = error <= self.tolerance
        return GateResult(
            passed=passed,
            score=error,
            reason=None if passed else "ctr_arithmetic_mismatch",
            metadata={"expected_ctr": expected_ctr, "reported_ctr": reported_ctr},
        )


class CardiomegalyRuleGate(StageGate):
    """Check final cardiomegaly decision against the accepted CTR threshold."""

    def __init__(
        self,
        default_threshold: float = 0.5,
        viewposition_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.default_threshold = default_threshold
        self.viewposition_thresholds = viewposition_thresholds or {"PA": 0.495, "AP": 0.545}

    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        if not isinstance(stage_output, dict):
            return GateResult(passed=False, reason="final_decision_not_structured")
        measurement = context.get("stage3_measurement") or context.get("measurement")
        if not isinstance(measurement, dict):
            return GateResult(passed=False, reason="missing_accepted_measurement")
        try:
            ctr = float(measurement["ctr"])
        except (KeyError, TypeError, ValueError) as exc:
            return GateResult(
                passed=False,
                reason="invalid_accepted_ctr",
                metadata={"error": str(exc)},
            )
        anatomy = context.get("stage2_anatomy") or {}
        viewposition = str(anatomy.get("viewposition") or stage_output.get("viewposition", "")).upper()
        threshold = self.viewposition_thresholds.get(viewposition, self.default_threshold)
        expected = ctr >= threshold
        if "cardiomegaly" not in stage_output:
            return GateResult(passed=False, reason="missing_final_decision")
        decision = bool(stage_output["cardiomegaly"])
        passed = decision == expected
        return GateResult(
            passed=passed,
            reason=None if passed else "decision_does_not_follow_ctr_threshold",
            metadata={"threshold": threshold, "expected_cardiomegaly": expected},
        )

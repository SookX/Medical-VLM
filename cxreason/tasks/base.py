from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from cxreason.controller.controller import StageSpec
from cxreason.gates.base import StageGate
from cxreason.gates.structured import (
    BinaryDecisionFieldGate,
    CompositeGate,
    CriterionTermsGate,
    MinMaxRatioConsistencyGate,
    RatioConsistencyGate,
    RequiredFieldsGate,
)


@dataclass(frozen=True)
class RatioCheck:
    target: str
    numerator: str
    denominator: str
    tolerance: float = 0.01

    def build_gate(self) -> StageGate:
        return RatioConsistencyGate(
            target=self.target,
            numerator=self.numerator,
            denominator=self.denominator,
            tolerance=self.tolerance,
        )


@dataclass(frozen=True)
class MinMaxRatioCheck:
    target: str
    value_a: str
    value_b: str
    tolerance: float = 0.01

    def build_gate(self) -> StageGate:
        return MinMaxRatioConsistencyGate(
            target=self.target,
            value_a=self.value_a,
            value_b=self.value_b,
            tolerance=self.tolerance,
        )


@dataclass(frozen=True)
class TaskSpec:
    name: str
    display_name: str
    preferred_criterion: str
    criterion_terms: tuple[str, ...]
    stage2_fields: tuple[str, ...]
    stage3_fields: tuple[str, ...]
    final_field: str
    stage2_numeric_fields: tuple[str, ...] = ()
    stage3_numeric_fields: tuple[str, ...] = ()
    ratio_checks: tuple[RatioCheck, ...] = ()
    minmax_ratio_checks: tuple[MinMaxRatioCheck, ...] = ()
    stage3_extra_gates: tuple[StageGate, ...] = ()
    final_gate: StageGate | None = None
    notes: str | None = None
    stage_names: tuple[str, ...] = field(
        default=("stage1_criterion", "stage2_anatomy", "stage3_measurement", "stage4_final")
    )

    @property
    def verification_levels(self) -> dict[str, str]:
        has_arithmetic = bool(self.ratio_checks or self.minmax_ratio_checks)
        has_sanity = bool(self.stage3_extra_gates)
        if has_arithmetic and has_sanity:
            stage3_level = "arithmetic+sanity"
        elif has_arithmetic:
            stage3_level = "arithmetic"
        elif has_sanity:
            stage3_level = "sanity"
        else:
            stage3_level = "structural"
        stage4_level = "clinical_rule" if self.final_gate is not None else "structural"
        return {
            "stage1_criterion": "criterion_terms",
            "stage2_anatomy": "structural",
            "stage3_measurement": stage3_level,
            "stage4_final": stage4_level,
        }

    def build_stage_specs(self) -> list[StageSpec]:
        measurement_gates: list[StageGate] = [
            RequiredFieldsGate(
                self.stage3_fields,
                numeric_fields=self.stage3_numeric_fields,
                stage_label="stage3_measurement",
            )
        ]
        measurement_gates.extend(check.build_gate() for check in self.ratio_checks)
        measurement_gates.extend(check.build_gate() for check in self.minmax_ratio_checks)
        measurement_gates.extend(self.stage3_extra_gates)

        return [
            StageSpec("stage1_criterion", CriterionTermsGate(self.criterion_terms)),
            StageSpec(
                "stage2_anatomy",
                RequiredFieldsGate(
                    self.stage2_fields,
                    numeric_fields=self.stage2_numeric_fields,
                    stage_label="stage2_anatomy",
                ),
            ),
            StageSpec("stage3_measurement", CompositeGate(measurement_gates)),
            StageSpec(
                "stage4_final",
                self.final_gate or BinaryDecisionFieldGate(self.final_field),
            ),
        ]


def tuple_from(fields: Iterable[str]) -> tuple[str, ...]:
    return tuple(fields)

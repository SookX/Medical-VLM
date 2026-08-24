from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cxreason.controller.state import ControllerState
from cxreason.gates.base import GateResult
from cxreason.tasks.base import TaskSpec


@dataclass(frozen=True)
class DependencyAuditResult:
    passed: bool
    reason: str | None = None
    stage_results: dict[str, GateResult] | None = None
    metadata: dict[str, Any] | None = None


def audit_accepted_chain(state: ControllerState, spec: TaskSpec) -> DependencyAuditResult:
    """Re-verify the accepted Path-1 chain with each stage's upstream context."""

    missing = [stage for stage in spec.stage_names if stage not in state.accepted_outputs]
    if missing:
        return DependencyAuditResult(
            passed=False,
            reason="missing_accepted_stages",
            metadata={"missing": missing},
        )

    stage_results: dict[str, GateResult] = {}
    context: dict[str, Any] = {}
    for stage_spec in spec.build_stage_specs():
        output = state.accepted_outputs[stage_spec.name]
        result = stage_spec.gate.verify(output, context)
        stage_results[stage_spec.name] = result
        if not result.passed:
            return DependencyAuditResult(
                passed=False,
                reason="accepted_stage_failed_reverification",
                stage_results=stage_results,
                metadata={"stage": stage_spec.name},
            )
        context[stage_spec.name] = output

    return DependencyAuditResult(passed=True, stage_results=stage_results)

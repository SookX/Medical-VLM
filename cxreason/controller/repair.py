from __future__ import annotations

from cxreason.gates.base import GateResult


def create_repair_feedback(stage: str, gate_result: GateResult) -> str:
    reason = gate_result.reason or "stage_failed_verification"
    return (
        f"Stage {stage} could not be verified. Reason: {reason}. "
        "Re-evaluate only this stage using the accepted upstream context. "
        "Do not assume or reveal any benchmark answer."
    )


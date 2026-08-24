from __future__ import annotations

from typing import Any

from cxreason.gates.structured import CompositeGate
from cxreason.tasks.base import TaskSpec
from cxreason.tasks.registry import TASK_REGISTRY


def _gate_names(gate: Any) -> list[str]:
    if isinstance(gate, CompositeGate):
        return [subgate.__class__.__name__ for subgate in gate.gates]
    return [gate.__class__.__name__]


def task_coverage_rows() -> list[dict[str, Any]]:
    rows = []
    for task, spec in TASK_REGISTRY.items():
        stage_specs = {stage.name: stage for stage in spec.build_stage_specs()}
        levels = spec.verification_levels
        rows.append(
            {
                "task": task,
                "display_name": spec.display_name,
                "stage1_level": levels["stage1_criterion"],
                "stage2_level": levels["stage2_anatomy"],
                "stage3_level": levels["stage3_measurement"],
                "stage4_level": levels["stage4_final"],
                "stage2_fields": list(spec.stage2_fields),
                "stage3_fields": list(spec.stage3_fields),
                "stage3_gates": _gate_names(stage_specs["stage3_measurement"].gate),
                "stage4_gates": _gate_names(stage_specs["stage4_final"].gate),
                "needs_oracle": True,
                "needs_practical_visual_verifier": True,
                "notes": spec.notes,
            }
        )
    return rows


def render_markdown_coverage(rows: list[dict[str, Any]] | None = None) -> str:
    rows = rows or task_coverage_rows()
    lines = [
        "# Task Coverage Report",
        "",
        "| Task | Stage 2 | Stage 3 | Stage 4 | Stage 3 Gates | Stage 4 Gates |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {task} | {stage2_level} | {stage3_level} | {stage4_level} | {stage3_gates} | {stage4_gates} |".format(
                task=row["task"],
                stage2_level=row["stage2_level"],
                stage3_level=row["stage3_level"],
                stage4_level=row["stage4_level"],
                stage3_gates=", ".join(row["stage3_gates"]),
                stage4_gates=", ".join(row["stage4_gates"]),
            )
        )
    lines.extend(
        [
            "",
            "All tasks still need oracle/reference gates for official CXReasonBench scoring.",
            "All tasks still need independent or explicitly benchmark-linked visual verifiers before visual grounding claims.",
            "",
        ]
    )
    return "\n".join(lines)

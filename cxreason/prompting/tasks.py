from __future__ import annotations

import json
from typing import Any

from cxreason.tasks.registry import get_task_spec


def _json_type_for_field(field: str) -> str:
    if field in {"required_structures"}:
        return "[string]"
    if field in {"direction", "direction_per_pnt", "viewposition", "task", "rule"}:
        return "string"
    return "number | string | null"


def _schema(fields: tuple[str, ...]) -> str:
    return json.dumps({field: _json_type_for_field(field) for field in fields}, indent=2)


def build_task_stage_prompt(
    task: str,
    stage: str,
    accepted_outputs: dict[str, Any],
    repair_feedback: str | None = None,
    development_context: dict[str, Any] | None = None,
) -> str:
    spec = get_task_spec(task)
    context = json.dumps(accepted_outputs, indent=2, sort_keys=True)
    repair = f"\nRepair feedback:\n{repair_feedback}\n" if repair_feedback else ""
    dev = ""
    if development_context:
        dev = (
            "\nDevelopment-only structured context. This is for JSON compliance dry runs, "
            "not benchmark evaluation:\n"
            f"{json.dumps(development_context, indent=2, sort_keys=True)}\n"
        )

    if stage == "stage1_criterion":
        instruction = (
            f"Stage 1: Select the diagnostic criterion for {spec.display_name}. "
            "Return only JSON with schema:\n"
            "{\n  \"criterion\": \"string\"\n}"
        )
    elif stage == "stage2_anatomy":
        instruction = (
            f"Stage 2: Identify the anatomical evidence needed for {spec.display_name}. "
            "Return only JSON with exactly these keys:\n"
            f"{_schema(spec.stage2_fields)}"
        )
    elif stage == "stage3_measurement":
        instruction = (
            f"Stage 3: Produce the measurement or recognition state for {spec.display_name}. "
            "Return only JSON with exactly these keys:\n"
            f"{_schema(spec.stage3_fields)}"
        )
    elif stage == "stage4_final":
        instruction = (
            f"Stage 4: Apply the clinical rule for {spec.display_name} using the accepted "
            "measurement state. Return only JSON with schema:\n"
            "{\n"
            f"  \"{spec.final_field}\": \"boolean\",\n"
            "  \"rule\": \"string\"\n"
            "}"
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return (
        "You are executing one CXReasonBench Path-1 stage. "
        "Use only the current image if provided and the accepted upstream context. "
        "Do not mention benchmark answers. Do not solve downstream stages early. "
        "Return only valid JSON.\n\n"
        f"Task: {task}\n"
        f"{instruction}\n"
        f"{repair}"
        f"\nAccepted upstream context:\n{context if accepted_outputs else '{}'}\n"
        f"{dev}"
    )


def build_all_task_prompt_templates() -> dict[str, dict[str, str]]:
    from cxreason.tasks.registry import list_task_names

    return {
        task: {
            stage: build_task_stage_prompt(task, stage, accepted_outputs={})
            for stage in get_task_spec(task).stage_names
        }
        for task in list_task_names()
    }


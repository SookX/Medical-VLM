from __future__ import annotations

import json
from typing import Any


def build_cardiomegaly_stage_prompt(
    stage: str,
    accepted_outputs: dict[str, Any],
    repair_feedback: str | None = None,
) -> str:
    context = json.dumps(accepted_outputs, indent=2, sort_keys=True)
    repair = f"\nRepair feedback:\n{repair_feedback}\n" if repair_feedback else ""

    if stage == "stage1_criterion":
        instruction = (
            "Stage 1: Select the diagnostic criterion for evaluating cardiomegaly. "
            "Return only JSON with this schema: {\"criterion\": string}."
        )
    elif stage == "stage2_anatomy":
        instruction = (
            "Stage 2: Identify the anatomical evidence needed for the selected criterion. "
            "Return only JSON with this schema: "
            "{\"required_structures\": [string], \"heart_xmin\": number, "
            "\"heart_xmax\": number, \"lung_xmin\": number, \"lung_xmax\": number, "
            "\"viewposition\": \"PA\" | \"AP\" | \"unknown\"}. "
            "If exact pixel coordinates cannot be estimated, use null for the coordinate."
        )
    elif stage == "stage3_measurement":
        instruction = (
            "Stage 3: Compute the cardiothoracic measurement from the accepted anatomy. "
            "Return only JSON with this schema: "
            "{\"heart_width\": number, \"lung_width\": number, \"ctr\": number}. "
            "CTR must equal heart_width divided by lung_width."
        )
    elif stage == "stage4_final":
        instruction = (
            "Stage 4: Apply the cardiomegaly rule to the accepted measurement. "
            "For development, use PA threshold 0.495 and AP threshold 0.545. "
            "Return only JSON with this schema: "
            "{\"cardiomegaly\": boolean, \"viewposition\": \"PA\" | \"AP\" | "
            "\"unknown\", \"rule\": string}."
        )
    else:
        raise ValueError(f"Unknown cardiomegaly stage: {stage}")

    return (
        "You are executing one CXReasonBench Path-1 cardiomegaly stage. "
        "Use only the image and accepted upstream context. "
        "Do not mention benchmark answers. Do not solve downstream stages early.\n\n"
        f"{instruction}\n"
        f"{repair}"
        f"\nAccepted upstream context:\n{context if accepted_outputs else '{}'}"
    )


from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from cxreason.controller.state import ControllerState


DEFAULT_STAGE_NAMES = (
    "stage1_criterion",
    "stage2_anatomy",
    "stage3_measurement",
    "stage4_final",
)


def reasoning_depth(state: ControllerState, stage_names: Iterable[str] = DEFAULT_STAGE_NAMES) -> int:
    depth = 0
    for stage in stage_names:
        if stage not in state.accepted_outputs:
            break
        depth += 1
    return depth


def state_metrics(
    state: ControllerState,
    stage_names: Iterable[str] = DEFAULT_STAGE_NAMES,
) -> dict[str, Any]:
    stage_names = tuple(stage_names)
    first_attempt_passed = {
        stage: bool(state.gate_results.get(stage) and state.gate_results[stage][0].passed)
        for stage in stage_names
        if stage in state.gate_results
    }
    post_repair_passed = {
        stage: stage in state.accepted_outputs
        for stage in stage_names
        if stage in state.gate_results
    }
    repaired_stages = [
        stage
        for stage in stage_names
        if len(state.gate_results.get(stage, [])) > 1
        and not state.gate_results[stage][0].passed
        and stage in state.accepted_outputs
    ]
    return {
        "case_passed": state.passed,
        "failed_stage": state.failed_stage,
        "reasoning_depth": reasoning_depth(state, stage_names),
        "model_calls": state.model_calls,
        "verifier_calls": state.verifier_calls,
        "generated_tokens": state.generated_tokens,
        "attempts": dict(state.attempts),
        "first_attempt_passed": first_attempt_passed,
        "post_repair_passed": post_repair_passed,
        "repaired_stages": repaired_stages,
    }


def run_record_from_state(
    state: ControllerState,
    *,
    system: str,
    run_id: str,
    mode: str,
    stage_names: Iterable[str] = DEFAULT_STAGE_NAMES,
    corrupted_stage: str | None = None,
    verification_level: str | None = None,
    dependency_audit_passed: bool | None = None,
    dependency_audit_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = state_metrics(state, stage_names)

    if corrupted_stage is None:
        first_attempt_failures = [
            stage
            for stage, results in state.gate_results.items()
            if results and not results[0].passed
        ]
        detected = False
        false_repair = bool(first_attempt_failures)
        repair_success = False
    else:
        first_result = state.gate_results.get(corrupted_stage, [None])[0]
        detected = bool(first_result and not first_result.passed)
        false_repair = False
        repair_success = bool(detected and state.passed)

    record = {
        "run_id": run_id,
        "system": system,
        "task": state.task,
        "case_id": state.case_id,
        "mode": mode,
        "corrupted_stage": corrupted_stage,
        "verification_level": verification_level,
        "detected": detected,
        "repair_success": repair_success,
        "false_repair": false_repair,
        "dependency_audit_passed": dependency_audit_passed,
        "dependency_audit_reason": dependency_audit_reason,
        **metrics,
    }
    if extra:
        record.update(extra)
    return record


def summarize_run_records(
    records: list[dict[str, Any]],
    group_fields: tuple[str, ...] = (
        "system",
        "task",
        "mode",
        "corrupted_stage",
        "verification_level",
    ),
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "cases": 0,
            "detected": 0,
            "completed": 0,
            "repair_success": 0,
            "false_repairs": 0,
            "dependency_audit_passed": 0,
            "model_calls": 0,
            "verifier_calls": 0,
            "generated_tokens": 0,
            "reasoning_depth": 0,
        }
    )

    for record in records:
        key = tuple(record.get(field) or "none" for field in group_fields)
        group = groups[key]
        group["cases"] += 1
        group["detected"] += int(record.get("detected", False))
        group["completed"] += int(record.get("case_passed", False))
        group["repair_success"] += int(record.get("repair_success", False))
        group["false_repairs"] += int(record.get("false_repair", False))
        group["dependency_audit_passed"] += int(record.get("dependency_audit_passed", False))
        group["model_calls"] += int(record.get("model_calls", 0))
        group["verifier_calls"] += int(record.get("verifier_calls", 0))
        group["generated_tokens"] += int(record.get("generated_tokens", 0))
        group["reasoning_depth"] += int(record.get("reasoning_depth", 0))

    by_group = []
    for key, group in sorted(groups.items()):
        cases = group["cases"]
        detected = group["detected"]
        row = {field: value for field, value in zip(group_fields, key)}
        row.update(group)
        row.update(
            {
                "detection_rate": detected / cases if cases else 0.0,
                "completion_rate": group["completed"] / cases if cases else 0.0,
                "repair_success_rate": group["repair_success"] / detected if detected else 0.0,
                "false_repair_rate": group["false_repairs"] / cases if cases else 0.0,
                "dependency_audit_pass_rate": (
                    group["dependency_audit_passed"] / cases if cases else 0.0
                ),
                "mean_model_calls": group["model_calls"] / cases if cases else 0.0,
                "mean_verifier_calls": group["verifier_calls"] / cases if cases else 0.0,
                "mean_generated_tokens": group["generated_tokens"] / cases if cases else 0.0,
                "mean_reasoning_depth": group["reasoning_depth"] / cases if cases else 0.0,
            }
        )
        by_group.append(row)

    overall_cases = len(records)
    return {
        "overall": {
            "cases": overall_cases,
            "detected": sum(int(record.get("detected", False)) for record in records),
            "completed": sum(int(record.get("case_passed", False)) for record in records),
            "false_repairs": sum(int(record.get("false_repair", False)) for record in records),
            "dependency_audit_passed": sum(
                int(record.get("dependency_audit_passed", False)) for record in records
            ),
            "model_calls": sum(int(record.get("model_calls", 0)) for record in records),
            "verifier_calls": sum(int(record.get("verifier_calls", 0)) for record in records),
        },
        "by_group": by_group,
    }

"""Validated task/stage support gates for Path 1 experiment runners."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from cxreason.data.cxreasonbench import PATH1_TASKS


PATH1_STAGE_ORDER = ("stage1", "stage1.5", "stage2", "stage3", "stage4")
REPAIRABLE_SUPPORT_LEVELS = frozenset({"validated_engineering", "frozen"})
KNOWN_SUPPORT_LEVELS = frozenset(
    {"unsupported", "rejected", "validated_engineering", "frozen"}
)
STAGE_ALIASES = {
    "criteria": "stage1",
    "custom_criteria": "stage1.5",
    "bodypart": "stage2",
    "measurement": "stage3",
    "final": "stage4",
}


class SupportPolicyError(ValueError):
    """Raised when a support policy or requested repair stage is unsafe."""


@dataclass(frozen=True)
class StageSupport:
    stage: str
    support_level: str
    repair_enabled: bool
    verifier: str | None
    evidence: str | None


@dataclass(frozen=True)
class TaskSupport:
    task: str
    matrix_status: str
    freeze_id: str | None
    stages: dict[str, StageSupport]

    @property
    def repair_stages(self) -> tuple[str, ...]:
        return tuple(
            stage for stage in PATH1_STAGE_ORDER if self.stages[stage].repair_enabled
        )


@dataclass(frozen=True)
class Path1TaskSupportPolicy:
    configuration_id: str
    tasks: dict[str, TaskSupport]
    diagnostic_all_1200_allowed: bool
    final_all_1200_allowed: bool
    final_ready_tasks: tuple[str, ...]
    blocking_reasons: tuple[str, ...]

    def task(self, task: str) -> TaskSupport:
        try:
            return self.tasks[task]
        except KeyError as exc:
            raise SupportPolicyError(f"Unknown Path 1 task: {task}") from exc

    def repair_stages(self, task: str) -> tuple[str, ...]:
        return self.task(task).repair_stages

    def validate_requested_stages(
        self, task: str, requested: Iterable[str]
    ) -> tuple[str, ...]:
        normalized: list[str] = []
        for label in requested:
            stage = STAGE_ALIASES.get(label, label)
            if stage not in PATH1_STAGE_ORDER:
                raise SupportPolicyError(f"Unknown repair stage: {label}")
            if stage not in normalized:
                normalized.append(stage)
        support = self.task(task)
        blocked = [stage for stage in normalized if not support.stages[stage].repair_enabled]
        if blocked:
            details = ", ".join(
                f"{stage}={support.stages[stage].support_level}" for stage in blocked
            )
            raise SupportPolicyError(
                f"Unsafe repair request for task {task}: {details}. "
                "Unsupported/rejected stages must pass through without repair."
            )
        return tuple(normalized)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SupportPolicyError(f"Expected a JSON object in {path}")
    return payload


def load_task_support_policy(path: Path) -> Path1TaskSupportPolicy:
    payload = _load_json(path)
    if payload.get("schema_version") != 1:
        raise SupportPolicyError("Unsupported task-support schema")
    if tuple(payload.get("task_order", ())) != PATH1_TASKS:
        raise SupportPolicyError("Task-support order does not match PATH1_TASKS")
    if tuple(payload.get("stage_order", ())) != PATH1_STAGE_ORDER:
        raise SupportPolicyError("Task-support stage order is invalid")
    defaults = payload.get("stage_defaults")
    task_payloads = payload.get("tasks")
    if not isinstance(defaults, dict) or set(defaults) != set(PATH1_STAGE_ORDER):
        raise SupportPolicyError("Every Path 1 stage requires a default policy")
    if not isinstance(task_payloads, dict) or set(task_payloads) != set(PATH1_TASKS):
        raise SupportPolicyError("Every Path 1 task requires an explicit policy")

    tasks: dict[str, TaskSupport] = {}
    for task in PATH1_TASKS:
        task_payload = task_payloads[task]
        overrides = task_payload.get("overrides", {})
        if not isinstance(overrides, dict) or not set(overrides).issubset(defaults):
            raise SupportPolicyError(f"Invalid stage overrides for {task}")
        stages: dict[str, StageSupport] = {}
        for stage in PATH1_STAGE_ORDER:
            merged = {**defaults[stage], **overrides.get(stage, {})}
            level = merged.get("support_level")
            enabled = merged.get("repair_enabled")
            verifier = merged.get("verifier")
            evidence = merged.get("evidence")
            if level not in KNOWN_SUPPORT_LEVELS:
                raise SupportPolicyError(f"Unknown support level for {task}/{stage}")
            if not isinstance(enabled, bool):
                raise SupportPolicyError(f"repair_enabled is not Boolean for {task}/{stage}")
            if enabled != (level in REPAIRABLE_SUPPORT_LEVELS):
                raise SupportPolicyError(
                    f"Unsafe support/repair combination for {task}/{stage}"
                )
            if enabled and (not verifier or not evidence):
                raise SupportPolicyError(
                    f"Enabled stage lacks verifier/evidence for {task}/{stage}"
                )
            if not enabled and level == "rejected" and not evidence:
                raise SupportPolicyError(
                    f"Rejected stage lacks retained evidence for {task}/{stage}"
                )
            stages[stage] = StageSupport(
                stage=stage,
                support_level=level,
                repair_enabled=enabled,
                verifier=verifier,
                evidence=evidence,
            )
        tasks[task] = TaskSupport(
            task=task,
            matrix_status=str(task_payload["matrix_status"]),
            freeze_id=task_payload.get("freeze_id"),
            stages=stages,
        )

    gates = payload.get("run_gates", {})
    final_ready = tuple(gates.get("final_ready_tasks", ()))
    if not set(final_ready).issubset(tasks):
        raise SupportPolicyError("Final-ready task set contains unknown tasks")
    for task in final_ready:
        if tasks[task].matrix_status != "frozen_full_path":
            raise SupportPolicyError(f"Final-ready task is not frozen: {task}")
    final_allowed = gates.get("final_all_1200_allowed")
    if final_allowed and set(final_ready) != set(PATH1_TASKS):
        raise SupportPolicyError("Full final run cannot be allowed with unready tasks")

    return Path1TaskSupportPolicy(
        configuration_id=str(payload["configuration_id"]),
        tasks=tasks,
        diagnostic_all_1200_allowed=bool(gates["diagnostic_all_1200_allowed"]),
        final_all_1200_allowed=bool(final_allowed),
        final_ready_tasks=final_ready,
        blocking_reasons=tuple(str(value) for value in gates["blocking_reasons"]),
    )

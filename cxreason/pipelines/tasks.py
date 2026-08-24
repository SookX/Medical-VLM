from __future__ import annotations

from cxreason.baselines.full_restart import FullRestartController
from cxreason.baselines.one_pass import OnePassController
from cxreason.baselines.self_consistency import SelfConsistencyController
from cxreason.controller.controller import LocalRepairController
from cxreason.tasks.registry import get_task_spec


def build_task_controller(task: str, max_attempts: int = 3) -> LocalRepairController:
    spec = get_task_spec(task)
    return LocalRepairController(
        stages=spec.build_stage_specs(),
        max_attempts_per_stage=max_attempts,
    )


def build_full_restart_task_controller(
    task: str,
    max_trajectories: int = 3,
) -> FullRestartController:
    spec = get_task_spec(task)
    return FullRestartController(
        stages=spec.build_stage_specs(),
        max_trajectories=max_trajectories,
    )


def build_one_pass_task_controller(task: str) -> OnePassController:
    spec = get_task_spec(task)
    return OnePassController(stages=spec.build_stage_specs())


def build_self_consistency_task_controller(
    task: str,
    candidates_per_stage: int = 3,
    generate_all_candidates: bool = False,
) -> SelfConsistencyController:
    spec = get_task_spec(task)
    return SelfConsistencyController(
        stages=spec.build_stage_specs(),
        candidates_per_stage=candidates_per_stage,
        generate_all_candidates=generate_all_candidates,
    )

from __future__ import annotations

from cxreason.controller.controller import LocalRepairController, StageSpec
from cxreason.gates.oracle import BenchmarkScorer, OracleGate
from cxreason.tasks.registry import get_task_spec


def build_oracle_task_controller(
    task: str,
    scorer: BenchmarkScorer,
    max_attempts: int = 3,
) -> LocalRepairController:
    spec = get_task_spec(task)
    return LocalRepairController(
        stages=[StageSpec(stage, OracleGate(stage, scorer)) for stage in spec.stage_names],
        max_attempts_per_stage=max_attempts,
    )

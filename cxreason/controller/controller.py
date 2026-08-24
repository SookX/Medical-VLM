from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from cxreason.controller.repair import create_repair_feedback
from cxreason.controller.state import ControllerState
from cxreason.gates.base import StageGate


class StageGenerator(Protocol):
    def generate(
        self,
        stage: str,
        accepted_outputs: dict[str, Any],
        repair_feedback: str | None = None,
    ) -> Any:
        ...


@dataclass(frozen=True)
class StageSpec:
    name: str
    gate: StageGate


class LocalRepairController:
    def __init__(self, stages: list[StageSpec], max_attempts_per_stage: int = 3) -> None:
        self.stages = stages
        self.max_attempts_per_stage = max_attempts_per_stage

    def run(self, case_id: str, task: str, generator: StageGenerator) -> ControllerState:
        state = ControllerState(case_id=case_id, task=task)

        for stage_spec in self.stages:
            repair_feedback = None
            accepted = False

            for _ in range(self.max_attempts_per_stage):
                output = generator.generate(
                    stage_spec.name,
                    state.accepted_outputs,
                    repair_feedback=repair_feedback,
                )
                gate_result = stage_spec.gate.verify(output, state.accepted_outputs)
                state.record_attempt(stage_spec.name, output, gate_result)

                if gate_result.passed:
                    state.accepted_outputs[stage_spec.name] = output
                    accepted = True
                    break

                repair_feedback = create_repair_feedback(stage_spec.name, gate_result)

            if not accepted:
                state.failed_stage = stage_spec.name
                break

        return state


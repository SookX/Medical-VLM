from __future__ import annotations

from cxreason.controller.controller import StageGenerator, StageSpec
from cxreason.controller.state import ControllerState


class OnePassController:
    """Generate each stage once and use gates only for evaluation."""

    def __init__(self, stages: list[StageSpec]) -> None:
        self.stages = stages

    def run(self, case_id: str, task: str, generator: StageGenerator) -> ControllerState:
        state = ControllerState(case_id=case_id, task=task)
        state.metadata["baseline"] = "one_pass"

        for stage_spec in self.stages:
            output = generator.generate(
                stage_spec.name,
                state.accepted_outputs,
                repair_feedback=None,
            )
            gate_result = stage_spec.gate.verify(output, state.accepted_outputs)
            state.record_attempt(stage_spec.name, output, gate_result)

            if not gate_result.passed:
                state.failed_stage = stage_spec.name
                break

            state.accepted_outputs[stage_spec.name] = output

        return state


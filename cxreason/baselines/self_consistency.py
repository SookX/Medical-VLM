from __future__ import annotations

from cxreason.controller.controller import StageGenerator, StageSpec
from cxreason.controller.state import ControllerState


class SelfConsistencyController:
    """Sample several candidates per stage and accept the first verified one."""

    def __init__(
        self,
        stages: list[StageSpec],
        candidates_per_stage: int = 3,
        generate_all_candidates: bool = False,
    ) -> None:
        self.stages = stages
        self.candidates_per_stage = candidates_per_stage
        self.generate_all_candidates = generate_all_candidates

    def run(self, case_id: str, task: str, generator: StageGenerator) -> ControllerState:
        state = ControllerState(case_id=case_id, task=task)
        state.metadata["baseline"] = "self_consistency"
        state.metadata["candidates_per_stage"] = self.candidates_per_stage
        state.metadata["generate_all_candidates"] = self.generate_all_candidates

        for stage_spec in self.stages:
            accepted_output = None

            for _ in range(self.candidates_per_stage):
                output = generator.generate(
                    stage_spec.name,
                    state.accepted_outputs,
                    repair_feedback=None,
                )
                gate_result = stage_spec.gate.verify(output, state.accepted_outputs)
                state.record_attempt(stage_spec.name, output, gate_result)

                if gate_result.passed and accepted_output is None:
                    accepted_output = output
                    if not self.generate_all_candidates:
                        break

            if accepted_output is None:
                state.failed_stage = stage_spec.name
                break

            state.accepted_outputs[stage_spec.name] = accepted_output

        return state


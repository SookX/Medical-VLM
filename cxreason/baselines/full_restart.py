from __future__ import annotations

from cxreason.controller.controller import StageGenerator, StageSpec
from cxreason.controller.state import ControllerState


class FullRestartController:
    """Restart the entire Path-1 chain when any stage is rejected."""

    def __init__(self, stages: list[StageSpec], max_trajectories: int = 3) -> None:
        self.stages = stages
        self.max_trajectories = max_trajectories

    def run(self, case_id: str, task: str, generator: StageGenerator) -> ControllerState:
        state = ControllerState(case_id=case_id, task=task)
        state.metadata["baseline"] = "full_restart"
        state.metadata["max_trajectories"] = self.max_trajectories

        for trajectory_idx in range(1, self.max_trajectories + 1):
            state.metadata["trajectory_attempts"] = trajectory_idx
            trajectory_outputs = {}
            failed_stage = None

            for stage_spec in self.stages:
                output = generator.generate(
                    stage_spec.name,
                    trajectory_outputs,
                    repair_feedback=None,
                )
                gate_result = stage_spec.gate.verify(output, trajectory_outputs)
                state.record_attempt(stage_spec.name, output, gate_result)

                if not gate_result.passed:
                    failed_stage = stage_spec.name
                    break

                trajectory_outputs[stage_spec.name] = output

            if failed_stage is None:
                state.accepted_outputs = trajectory_outputs
                return state

            if trajectory_idx == self.max_trajectories:
                state.accepted_outputs = trajectory_outputs
                state.failed_stage = failed_stage
                return state

        return state


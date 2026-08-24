from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cxreason.gates.base import GateResult


@dataclass
class ControllerState:
    case_id: str
    task: str
    initial_answer: Any | None = None
    stage_outputs: dict[str, list[Any]] = field(default_factory=dict)
    accepted_outputs: dict[str, Any] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    gate_results: dict[str, list[GateResult]] = field(default_factory=dict)
    model_calls: int = 0
    verifier_calls: int = 0
    generated_tokens: int = 0
    failed_stage: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.failed_stage is None

    def record_attempt(
        self,
        stage: str,
        output: Any,
        gate_result: GateResult,
        generated_tokens: int = 0,
    ) -> None:
        self.stage_outputs.setdefault(stage, []).append(output)
        self.gate_results.setdefault(stage, []).append(gate_result)
        self.attempts[stage] = self.attempts.get(stage, 0) + 1
        self.model_calls += 1
        self.verifier_calls += 1
        self.generated_tokens += generated_tokens

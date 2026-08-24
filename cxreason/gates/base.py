from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateResult:
    passed: bool
    score: float | None = None
    reason: str | None = None
    metadata: dict[str, Any] | None = None


class StageGate(ABC):
    @abstractmethod
    def verify(self, stage_output: Any, context: dict[str, Any]) -> GateResult:
        raise NotImplementedError


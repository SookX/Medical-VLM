"""Native Path-1 generator backed by the persistent MedGemma session."""

from __future__ import annotations

from cxreason.data.cxreasonbench import NativeStageTurn
from cxreason.modeling.medgemma_session import HistoryEntry, MedGemmaSession
from cxreason.path1.execution import NativeGeneration


class NativeMedGemmaGenerator:
    def __init__(
        self,
        session: MedGemmaSession,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        base_seed: int | None = None,
    ) -> None:
        if temperature > 0.0 and base_seed is None:
            raise ValueError("base_seed is required for sampled generation")
        self.session = session
        self.temperature = temperature
        self.top_p = top_p
        self.base_seed = base_seed
        self.live_calls = 0
        self.input_tokens = 0
        self.generated_tokens = 0

    def generate(
        self,
        turn: NativeStageTurn,
        accepted_history: list[HistoryEntry],
        repair_feedback: str | None = None,
    ) -> NativeGeneration:
        query = turn.question
        if repair_feedback:
            query = f"{query}\n\nVerification feedback: {repair_feedback}"
        generation_kwargs = {}
        if self.temperature > 0.0:
            generation_kwargs = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "sampling_seed": self.base_seed + self.live_calls,
            }
        result = self.session.generate_with_metadata(
            query=query,
            image_paths=list(turn.image_paths),
            history=accepted_history,
            **generation_kwargs,
        )
        self.live_calls += 1
        self.input_tokens += result.input_tokens
        self.generated_tokens += result.generated_tokens
        return NativeGeneration(
            text=result.text,
            generated_tokens=result.generated_tokens,
        )

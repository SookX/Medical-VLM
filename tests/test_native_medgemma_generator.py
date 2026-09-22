from __future__ import annotations

from pathlib import Path

import pytest

from cxreason.data.cxreasonbench import NativeStageTurn
from cxreason.generators.native_medgemma import NativeMedGemmaGenerator
from cxreason.modeling.medgemma_session import MedGemmaGeneration


class FakeSession:
    def __init__(self) -> None:
        self.calls = []

    def generate_with_metadata(self, **kwargs):
        self.calls.append(kwargs)
        return MedGemmaGeneration("FINAL ANSWER: (a)", 10, 3)


def _turn() -> NativeStageTurn:
    return NativeStageTurn(
        key="bodypart_0",
        scorer_stage="bodypart",
        question="Choose. Options: (a) First, (b) Second",
        answer="(a) First",
        image_paths=(Path("one.png"),),
        source_path=Path("question.json"),
    )


def test_greedy_generator_preserves_legacy_session_arguments() -> None:
    session = FakeSession()
    generator = NativeMedGemmaGenerator(session)
    generator.generate(_turn(), [])
    assert set(session.calls[0]) == {"query", "image_paths", "history"}


def test_sampled_generator_uses_reproducible_incrementing_seeds() -> None:
    session = FakeSession()
    generator = NativeMedGemmaGenerator(
        session, temperature=0.7, top_p=0.9, base_seed=100
    )
    generator.generate(_turn(), [])
    generator.generate(_turn(), [])
    assert [call["sampling_seed"] for call in session.calls] == [100, 101]
    assert all(call["temperature"] == 0.7 for call in session.calls)
    assert all(call["top_p"] == 0.9 for call in session.calls)


def test_sampled_generator_requires_seed() -> None:
    with pytest.raises(ValueError, match="base_seed"):
        NativeMedGemmaGenerator(FakeSession(), temperature=0.7)

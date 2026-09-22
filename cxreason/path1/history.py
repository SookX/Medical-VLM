"""Controlled representations of accepted answers in later Path-1 history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cxreason.data.cxreasonbench import NativeStageTurn
from cxreason.path1.practical_verifiers import (
    parse_mcq_options,
    selected_option_letters,
)


HistoryFormat = Literal["canonical", "option_text"]


def normalized_selected_response(
    question: str,
    response: str,
    *,
    mode: HistoryFormat,
) -> str:
    """Represent the same selected options without preserving model reasoning."""

    if mode not in {"canonical", "option_text"}:
        raise ValueError(f"Unknown accepted-history format: {mode!r}")
    letters = selected_option_letters(question, response)
    if not letters:
        raise ValueError("Cannot normalize a response with no selected option")
    if mode == "canonical":
        return f"FINAL ANSWER: ({', '.join(letters)})"

    options = parse_mcq_options(question)
    selected = [f"({letter}) {options[letter]}" for letter in letters]
    return "FINAL ANSWER: " + "; ".join(selected)


@dataclass(frozen=True)
class RepairedStageHistoryFormatter:
    """Normalize only accepted retry outputs from one stage group.

    Initial accepted answers and accepted answers from other stages remain byte-for-byte
    unchanged. This permits an ablation in which routing and retry generation are fixed,
    while only the downstream context after a successful repair changes.
    """

    stage_group: str
    mode: HistoryFormat

    def __post_init__(self) -> None:
        if not self.stage_group.strip():
            raise ValueError("stage_group must be non-empty")
        if self.mode not in {"canonical", "option_text"}:
            raise ValueError(f"Unknown accepted-history format: {self.mode!r}")

    def __call__(
        self,
        turn: NativeStageTurn,
        response: str,
        attempt_index: int,
        repair_enabled: bool,
    ) -> str:
        if (
            turn.scorer_stage != self.stage_group
            or attempt_index <= 1
            or not repair_enabled
        ):
            return response
        return normalized_selected_response(turn.question, response, mode=self.mode)

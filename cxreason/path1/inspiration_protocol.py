"""Answer-blind clinical protocol rules for the CXReasonBench Inspiration path."""

from __future__ import annotations

import re

from cxreason.data.cxreasonbench import NativeStageTurn
from cxreason.path1.practical_verifiers import (
    parse_mcq_options,
    selected_option_letters,
)


MINIMUM_ADEQUATE_POSTERIOR_RIBS = 9


def measurement_count(question: str, response: str) -> int | None:
    selected = selected_option_letters(question, response)
    options = parse_mcq_options(question)
    if len(selected) != 1 or selected[0] not in options:
        return None
    matches = re.findall(r"(?<!\d)(7|8|9|10|11)(?!\d)", options[selected[0]])
    return int(matches[0]) if len(matches) == 1 else None


def expected_final_letter(turn: NativeStageTurn, count: int) -> str | None:
    target = "good" if count >= MINIMUM_ADEQUATE_POSTERIOR_RIBS else "poor"
    matches = [
        letter
        for letter, option in parse_mcq_options(turn.question).items()
        if target in option.lower()
    ]
    return matches[0] if len(matches) == 1 else None

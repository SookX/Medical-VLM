"""Deterministic scoring helpers for CXReasonBench multiple-choice outputs.

The published evaluator delegates answer equivalence and numeric extraction to
Gemini 2.0 Flash. That endpoint is no longer available. These helpers provide
an auditable offline fallback that prioritizes explicit option letters and only
falls back to matching option text when a response omits the letter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_OPTION_RE = re.compile(
    r"\(([a-j])\)\s*(.*?)(?=(?:\s*\([a-j]\)\s*)|$)",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_MARKER_RE = re.compile(
    r"(?:final\s+answer\s*:|the\s+answer\s+is|answer\s*:)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChoiceScore:
    score: int
    selected: tuple[str, ...]
    reference: tuple[str, ...]
    method: str


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.]+", " ", text.lower()).split())


def _answer_tail(response: str) -> str:
    matches = list(_FINAL_MARKER_RE.finditer(response))
    if matches:
        return response[matches[-1].end() :].strip()
    return response.strip()


def _letters(text: str) -> tuple[str, ...]:
    letters: list[str] = []
    for contents in re.findall(r"\(([^()]*)\)", text):
        candidates = re.findall(r"\b([a-j])\b", contents, re.IGNORECASE)
        residue = re.sub(
            r"\b[a-j]\b|[,/&]|\band\b|\s+", "", contents, flags=re.IGNORECASE
        )
        if candidates and not residue:
            letters.extend(letter.lower() for letter in candidates)
    return tuple(dict.fromkeys(letters))


def _options(question: str) -> dict[str, str]:
    return {letter.lower(): value.strip(" .,\n") for letter, value in _OPTION_RE.findall(question)}


def _semantic_letters(question: str, response: str) -> tuple[str, ...]:
    """Infer selected letters from option text when no explicit letter exists."""

    options = _options(question)
    tail = _normalize(_answer_tail(response))
    if not tail:
        return ()

    matches: list[str] = []
    for letter, option in options.items():
        normalized = _normalize(option)
        if not normalized:
            continue
        if normalized in {"yes", "no", "i don t know", "none of the above"}:
            pattern = r"\b" + r"\s+".join(map(re.escape, normalized.split())) + r"\b"
            if re.search(pattern, tail):
                matches.append(letter)
        elif len(normalized) >= 8 and normalized in tail:
            matches.append(letter)
    return tuple(dict.fromkeys(matches))


def selected_letters(question: str, response: str) -> tuple[tuple[str, ...], str]:
    tail = _answer_tail(response)
    explicit = _letters(tail)
    if not explicit:
        explicit = tuple(
            dict.fromkeys(
                letter.lower()
                for letter in re.findall(
                    r"\b(?:option|choice)\s+([a-j])\b", tail, re.IGNORECASE
                )
            )
        )
    if explicit:
        return explicit, "letter"
    return _semantic_letters(question, response), "option_text"


def _special_letter(question: str, phrase: str) -> str | None:
    normalized_phrase = _normalize(phrase)
    for letter, option in _options(question).items():
        if normalized_phrase in _normalize(option):
            return letter
    return None


def score_choice(
    *,
    stage: str,
    question: str,
    answer: str,
    response: str,
) -> ChoiceScore:
    """Score one greedy response using upstream-compatible integer labels.

    Returns 1 for correct, 0 for incorrect, -1 for IDK, and -2 for N/A.
    """

    reference = _letters(answer)
    selected, method = selected_letters(question, response)
    normalized_tail = _normalize(_answer_tail(response))

    idk_letter = _special_letter(question, "I don't know")
    none_letter = _special_letter(question, "None of the above")

    if stage == "init" and (
        (idk_letter is not None and selected == (idk_letter,))
        or "i don t know" in normalized_tail
    ):
        return ChoiceScore(-1, selected, reference, method)

    if stage in {"criteria", "bodypart"} and (
        (none_letter is not None and none_letter in selected)
        or "none of the above" in normalized_tail
    ):
        return ChoiceScore(-2, selected, reference, method)

    if stage == "custom_criteria" and selected != reference:
        return ChoiceScore(-1, selected, reference, method)

    if stage == "final" and len(reference) > 1:
        correct = len(selected) == 1 and selected[0] in reference
    else:
        correct = bool(selected) and set(selected) == set(reference)

    return ChoiceScore(int(correct), selected, reference, method)


def extract_numbers(text: str) -> list[float]:
    no_ranges = re.sub(r"\d+\.?\d*\s*-\s*\d+\.?\d*", "", text)
    return [float(value) for value in re.findall(r"[+]?\d*\.\d+|\d+", no_ranges.replace(" ", ""))]


def measurement_matches(task: str, measurement_answer: str, final_response: str) -> bool:
    """Reproduce the numeric consistency test in upstream ``metric.py``."""

    answer_values = [
        float(value) for value in re.findall(r"[-+]?\d*\.\d+|\d+", measurement_answer)
    ]
    model_values = extract_numbers(final_response)
    if not answer_values:
        return False

    if task == "projection":
        if len(answer_values) < 4 or len(model_values) != 2:
            return False
        return (
            answer_values[0] <= model_values[0] <= answer_values[1]
            and answer_values[2] <= model_values[1] <= answer_values[3]
        )

    return (
        len(set(model_values)) == 1
        and answer_values[0] <= model_values[0] <= answer_values[-1]
    )


def aggregate_subscores(scores: Iterable[int]) -> int:
    values = tuple(scores)
    return int(bool(values) and len(values) == sum(values))

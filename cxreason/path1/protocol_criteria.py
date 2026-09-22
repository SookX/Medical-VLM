"""Answer-blind protocol-semantic selectors for Path-1 criterion stages."""

from __future__ import annotations

from dataclasses import dataclass

from cxreason.data.cxreasonbench import NativeStageTurn
from cxreason.gates.base import GateResult
from cxreason.path1.practical_verifiers import (
    _normalize,
    parse_mcq_options,
    selected_option_letters,
)


@dataclass(frozen=True)
class CriterionConcept:
    required: tuple[str, ...]
    any_of: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()

    def matches(self, option: str) -> bool:
        value = _normalize(option)
        return (
            all(term in value for term in self.required)
            and (not self.any_of or any(term in value for term in self.any_of))
            and not any(term in value for term in self.excluded)
        )


# These concepts come from the task definitions and clinical protocol, not from
# benchmark answer letters. They intentionally describe only direct criteria.
CRITERION_CONCEPTS: dict[str, CriterionConcept] = {
    "aortic_knob_enlargement": CriterionConcept(
        ("aortic knob",), ("bulge", "widening")
    ),
    "ascending_aorta_enlargement": CriterionConcept(
        ("ascending aorta",), ("bulge", "widening")
    ),
    "cardiomegaly": CriterionConcept(("cardiothoracic ratio",)),
    "carina_angle": CriterionConcept(("angle", "carina")),
    "descending_aorta_enlargement": CriterionConcept(
        ("descending aorta",), ("dilation", "widening"), ("tortuos",)
    ),
    "descending_aorta_tortuous": CriterionConcept(
        ("descending aorta",), ("tortuos", "bends", "twists")
    ),
    "inclusion": CriterionConcept(
        ("lung apices", "lateral ribs", "costophrenic angles")
    ),
    "inspiration": CriterionConcept(
        ("right posterior ribs", "right hemidiaphragm"), ("count", "counting")
    ),
    "mediastinal_widening": CriterionConcept(("width", "mediastinum")),
    "projection": CriterionConcept(
        ("scapula", "lung fields"), ("retract", "overlap")
    ),
    "rotation": CriterionConcept(
        ("spinous processes", "medial ends", "clavicles")
    ),
    "trachea_deviation": CriterionConcept(
        ("trachea", "midline"), ("displaced", "deviation")
    ),
}


REFINED_PROMPT_ANCHORS: dict[str, tuple[str, ...]] = {
    "aortic_knob_enlargement": ("aortic knob", "trachea"),
    "ascending_aorta_enlargement": ("ascending aorta", "right heart"),
    "descending_aorta_enlargement": ("descending aorta", "trachea"),
    "descending_aorta_tortuous": ("descending aorta", "curvature"),
    "inspiration": ("mid clavicular line", "right posterior ribs"),
    "mediastinal_widening": ("mediastinal width", "thoracic width"),
    "projection": ("scapula", "overlap ratio"),
}


def render_option_response(turn: NativeStageTurn, letter: str) -> str:
    options = parse_mcq_options(turn.question)
    if letter not in options:
        raise ValueError(f"Unknown option letter: {letter}")
    return f"FINAL ANSWER: ({letter}) {options[letter]}"


class ProtocolCriterionVerifier:
    """Select a task's direct criterion or its explicit continuation option."""

    name = "protocol_criterion_semantics_v1"

    def __init__(self, task: str) -> None:
        try:
            self.concept = CRITERION_CONCEPTS[task]
        except KeyError as exc:
            raise ValueError(f"Unsupported criterion task: {task}") from exc
        self.task = task

    def expected_letter(self, turn: NativeStageTurn) -> tuple[str | None, str]:
        if turn.scorer_stage != "criteria":
            return None, "wrong_stage"
        options = parse_mcq_options(turn.question)
        direct = [
            letter for letter, option in options.items() if self.concept.matches(option)
        ]
        continuation = [
            letter
            for letter, option in options.items()
            if "need new option" in _normalize(option)
        ]
        no_match = [
            letter
            for letter, option in options.items()
            if "none of the above" in _normalize(option) or _normalize(option) == "none"
        ]
        if len(direct) == 1:
            return direct[0], "direct_protocol_criterion"
        if not direct and len(continuation) == 1:
            return continuation[0], "request_refined_options"
        if not direct and not continuation and len(no_match) == 1:
            return no_match[0], "none_of_the_above"
        return None, "ambiguous_protocol_mapping"

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        del accepted_responses
        expected, mapping = self.expected_letter(turn)
        selected = selected_option_letters(turn.question, response)
        if expected is None:
            return GateResult(
                passed=True,
                metadata={
                    "verifier": self.name,
                    "task": self.task,
                    "decision": "ABSTAIN",
                    "abstain_reason": mapping,
                    "selected": list(selected),
                },
            )
        passed = selected == (expected,)
        return GateResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=None if passed else "selection_disagrees_with_protocol_criterion",
            metadata={
                "verifier": self.name,
                "task": self.task,
                "decision": "PASS" if passed else "REPAIR",
                "mapping": mapping,
                "expected_option": expected,
                "selected": list(selected),
            },
        )


class ProtocolRefinedCriterionVerifier:
    """Select Yes only for a recognized task-specific refinement prompt."""

    name = "protocol_refined_criterion_semantics_v1"

    def __init__(self, task: str) -> None:
        try:
            self.anchors = REFINED_PROMPT_ANCHORS[task]
        except KeyError as exc:
            raise ValueError(f"Task has no recognized refinement protocol: {task}") from exc
        self.task = task

    def expected_letter(self, turn: NativeStageTurn) -> tuple[str | None, str]:
        if turn.scorer_stage != "custom_criteria":
            return None, "wrong_stage"
        question = _normalize(turn.question)
        options = parse_mcq_options(turn.question)
        yes = [letter for letter, option in options.items() if _normalize(option) == "yes"]
        no = [
            letter
            for letter, option in options.items()
            if _normalize(option) == "no" or _normalize(option).startswith("no ")
        ]
        asks_applicability = (
            "can you apply this refined criterion" in question
            or "can you apply this refined method" in question
        )
        recognized = asks_applicability and all(anchor in question for anchor in self.anchors)
        if recognized and len(yes) == 1 and len(no) == 1:
            return yes[0], "accept_recognized_refinement"
        return None, "unrecognized_refinement_protocol"

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        del accepted_responses
        expected, mapping = self.expected_letter(turn)
        selected = selected_option_letters(turn.question, response)
        if expected is None:
            return GateResult(
                passed=True,
                metadata={
                    "verifier": self.name,
                    "task": self.task,
                    "decision": "ABSTAIN",
                    "abstain_reason": mapping,
                    "selected": list(selected),
                },
            )
        passed = selected == (expected,)
        return GateResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=None if passed else "selection_disagrees_with_refined_protocol",
            metadata={
                "verifier": self.name,
                "task": self.task,
                "decision": "PASS" if passed else "REPAIR",
                "mapping": mapping,
                "expected_option": expected,
                "selected": list(selected),
            },
        )

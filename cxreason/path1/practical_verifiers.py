"""Answer-independent consistency verifiers for native Path 1 outputs."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path

from cxreason.data.cxreasonbench import NativePath1Case, NativeStageTurn
from cxreason.gates.base import GateResult


_FINAL_MARKER_RE = re.compile(
    r"(?:final\s+answer\s*:|the\s+answer\s+is|answer\s*:)", re.IGNORECASE
)
_OPTION_MARKER_RE = re.compile(r"\(([a-j])\)\s*", re.IGNORECASE)
_RANGE_RE = re.compile(
    r"\[?\s*(-?\d+(?:\.\d+)?)\s*[-\u2013\u2014]\s*"
    r"(-?\d+(?:\.\d+)?)\s*\]?"
)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.]+", " ", text.lower()).split())


def _answer_tail(response: str) -> str:
    matches = list(_FINAL_MARKER_RE.finditer(response))
    return response[matches[-1].end() :].strip() if matches else response.strip()


def parse_mcq_options(question: str) -> dict[str, str]:
    """Parse native options through ``(j)`` without using reference answers."""

    matches = list(_OPTION_MARKER_RE.finditer(question))
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        options[match.group(1).lower()] = question[match.end() : end].strip(" .,\n")
    return options


def selected_option_letters(question: str, response: str) -> tuple[str, ...]:
    """Extract one or more selected native option letters through ``j``."""

    options = parse_mcq_options(question)
    tail = _answer_tail(response)
    letters: list[str] = []
    for contents in re.findall(r"\(([^()]*)\)", tail):
        candidates = re.findall(r"\b([a-j])\b", contents, re.IGNORECASE)
        residue = re.sub(
            r"\b[a-j]\b|[,/&]|\band\b|\s+", "", contents, flags=re.IGNORECASE
        )
        if candidates and not residue:
            letters.extend(letter.lower() for letter in candidates)
    if not letters:
        letters.extend(
            letter.lower()
            for letter in re.findall(
                r"\b(?:option|choice)\s+([a-j])\b", tail, re.IGNORECASE
            )
        )
    if not letters:
        normalized_tail = _normalize(tail)
        for letter, option in options.items():
            normalized_option = _normalize(option)
            if normalized_option and normalized_option in normalized_tail:
                letters.append(letter)
    return tuple(dict.fromkeys(letter for letter in letters if letter in options))


def _numeric_interval(option: str) -> tuple[float, float] | None:
    match = _RANGE_RE.search(option)
    if match is None:
        return None
    low, high = float(match.group(1)), float(match.group(2))
    return (min(low, high), max(low, high))


def _possible_threshold(
    interval: tuple[float, float], threshold: float, operator: str
) -> frozenset[bool]:
    low, high = interval
    if operator == ">=":
        if low >= threshold:
            return frozenset({True})
        if high < threshold:
            return frozenset({False})
    elif operator == ">":
        if low > threshold:
            return frozenset({True})
        if high <= threshold:
            return frozenset({False})
    elif operator == "<=":
        if high <= threshold:
            return frozenset({True})
        if low > threshold:
            return frozenset({False})
    elif operator == "<":
        if high < threshold:
            return frozenset({True})
        if low >= threshold:
            return frozenset({False})
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    return frozenset({False, True})


def _possible_in_range(
    interval: tuple[float, float], normal_low: float, normal_high: float
) -> frozenset[bool]:
    low, high = interval
    if low >= normal_low and high <= normal_high:
        return frozenset({True})
    if high < normal_low or low > normal_high:
        return frozenset({False})
    return frozenset({False, True})


@dataclass(frozen=True)
class FinalDecisionInference:
    possible_positive: frozenset[bool]
    reason: str
    evidence: dict[str, object]


class PracticalFinalConsistencyVerifier:
    """Check Stage 4 against accepted Stage 3 evidence and the stated rule.

    The verifier receives question text and prior model responses only. It is
    conservative: an interval that straddles a decision boundary causes an
    abstention (implemented as PASS), not a repair.
    """

    name = "practical_final_consistency_v1"

    _NUMERIC_POLICIES: dict[str, tuple[float, str, bool]] = {
        "aortic_knob_enlargement": (2.5, ">=", True),
        "descending_aorta_enlargement": (2.5, ">=", True),
        "descending_aorta_tortuous": (0.0009, ">=", True),
        "rotation": (0.4, ">", False),
    }

    def __init__(
        self,
        *,
        task: str,
        initial_question: str,
        measurement_question: str,
    ) -> None:
        self.task = task
        self.initial_question = initial_question
        self.measurement_question = measurement_question

    @classmethod
    def from_case(cls, case: NativePath1Case) -> "PracticalFinalConsistencyVerifier":
        return cls(
            task=case.task,
            initial_question=case.initial.question,
            measurement_question=case.measurement.question,
        )

    def _selected_measurement_options(
        self, response: str
    ) -> tuple[tuple[str, str], ...]:
        options = parse_mcq_options(self.measurement_question)
        letters = selected_option_letters(self.measurement_question, response)
        return tuple((letter, options[letter]) for letter in letters if letter in options)

    def _numeric_inference(
        self,
        selected: tuple[tuple[str, str], ...],
        *,
        threshold: float,
        operator: str,
        positive_when_predicate: bool,
    ) -> FinalDecisionInference:
        if len(selected) != 1:
            return FinalDecisionInference(frozenset(), "measurement_selection_not_single", {})
        letter, option = selected[0]
        interval = _numeric_interval(option)
        if interval is None:
            return FinalDecisionInference(
                frozenset(), "measurement_interval_not_found", {"letter": letter}
            )
        possible = _possible_threshold(interval, threshold, operator)
        if not positive_when_predicate:
            possible = frozenset(not value for value in possible)
        return FinalDecisionInference(
            possible,
            "numeric_threshold",
            {
                "measurement_letter": letter,
                "interval": list(interval),
                "threshold": threshold,
                "operator": operator,
                "positive_when_predicate": positive_when_predicate,
            },
        )

    def infer(self, measurement_response: str) -> FinalDecisionInference:
        selected = self._selected_measurement_options(measurement_response)

        if self.task in self._NUMERIC_POLICIES:
            threshold, operator, positive_when = self._NUMERIC_POLICIES[self.task]
            return self._numeric_inference(
                selected,
                threshold=threshold,
                operator=operator,
                positive_when_predicate=positive_when,
            )

        if self.task in {"cardiomegaly", "mediastinal_widening"}:
            view_match = re.search(r"\b(AP|PA)\s*\(", self.initial_question, re.IGNORECASE)
            if view_match is None:
                return FinalDecisionInference(frozenset(), "view_position_not_found", {})
            view = view_match.group(1).upper()
            if self.task == "cardiomegaly":
                threshold = 0.55 if view == "AP" else 0.50
            else:
                threshold = 0.33 if view == "AP" else 0.28
            inferred = self._numeric_inference(
                selected,
                threshold=threshold,
                operator=">=",
                positive_when_predicate=True,
            )
            return FinalDecisionInference(
                inferred.possible_positive,
                inferred.reason,
                {**inferred.evidence, "view_position": view},
            )

        if self.task == "carina_angle":
            if len(selected) != 1:
                return FinalDecisionInference(
                    frozenset(), "measurement_selection_not_single", {}
                )
            letter, option = selected[0]
            interval = _numeric_interval(option)
            if interval is None:
                return FinalDecisionInference(
                    frozenset(), "measurement_interval_not_found", {"letter": letter}
                )
            return FinalDecisionInference(
                _possible_in_range(interval, 40.0, 80.0),
                "normal_range",
                {
                    "measurement_letter": letter,
                    "interval": list(interval),
                    "normal_range": [40.0, 80.0],
                },
            )

        if self.task == "inspiration":
            if len(selected) != 1:
                return FinalDecisionInference(
                    frozenset(), "measurement_selection_not_single", {}
                )
            letter, option = selected[0]
            match = re.search(r"\b(\d+)\b", option)
            if match is None:
                return FinalDecisionInference(
                    frozenset(), "rib_count_not_found", {"letter": letter}
                )
            rib_count = int(match.group(1))
            return FinalDecisionInference(
                frozenset({rib_count >= 9}),
                "rib_count_threshold",
                {"measurement_letter": letter, "rib_count": rib_count, "threshold": 9},
            )

        if self.task == "ascending_aorta_enlargement":
            if len(selected) != 1:
                return FinalDecisionInference(
                    frozenset(), "measurement_selection_not_single", {}
                )
            letter, option = selected[0]
            normalized = _normalize(option)
            if normalized.startswith("does not extend beyond"):
                positive = False
            elif normalized.startswith("extends beyond"):
                positive = True
            else:
                return FinalDecisionInference(
                    frozenset(), "extension_label_not_understood", {"letter": letter}
                )
            return FinalDecisionInference(
                frozenset({positive}),
                "extension_label",
                {"measurement_letter": letter, "measurement_option": option},
            )

        if self.task == "inclusion":
            if len(selected) != 1:
                return FinalDecisionInference(
                    frozenset(), "measurement_selection_not_single", {}
                )
            letter, option = selected[0]
            normalized = _normalize(option)
            if "included" not in normalized and "excluded" not in normalized:
                return FinalDecisionInference(
                    frozenset(), "inclusion_label_not_understood", {"letter": letter}
                )
            positive = "excluded" not in normalized and normalized.count("included") >= 6
            return FinalDecisionInference(
                frozenset({positive}),
                "all_regions_included",
                {"measurement_letter": letter, "measurement_option": option},
            )

        if self.task == "trachea_deviation":
            if len(selected) != 1:
                return FinalDecisionInference(
                    frozenset(), "measurement_selection_not_single", {}
                )
            letter, option = selected[0]
            normalized = _normalize(option)
            if "deviat" not in normalized:
                return FinalDecisionInference(
                    frozenset(), "deviation_label_not_understood", {"letter": letter}
                )
            return FinalDecisionInference(
                frozenset({not normalized.startswith("not deviated")}),
                "deviation_label",
                {"measurement_letter": letter, "measurement_option": option},
            )

        if self.task == "projection":
            if len(selected) != 2:
                return FinalDecisionInference(
                    frozenset(), "projection_requires_two_measurements", {}
                )
            right = [item for item in selected if _normalize(item[1]).startswith("right")]
            left = [item for item in selected if _normalize(item[1]).startswith("left")]
            if len(right) != 1 or len(left) != 1:
                return FinalDecisionInference(
                    frozenset(), "projection_sides_not_identified", {}
                )
            intervals = [_numeric_interval(right[0][1]), _numeric_interval(left[0][1])]
            if any(interval is None for interval in intervals):
                return FinalDecisionInference(
                    frozenset(), "projection_interval_not_found", {}
                )
            possible_sides = [
                _possible_threshold(interval, 0.3, "<")
                for interval in intervals
                if interval is not None
            ]
            possible_pa = frozenset(
                all(values) for values in itertools.product(*possible_sides)
            )
            return FinalDecisionInference(
                possible_pa,
                "bilateral_projection_threshold",
                {
                    "measurement_letters": [right[0][0], left[0][0]],
                    "intervals": [list(interval) for interval in intervals if interval],
                    "threshold": 0.3,
                },
            )

        return FinalDecisionInference(
            frozenset(), "unsupported_task", {"task": self.task}
        )

    def _final_polarity(self, question: str) -> tuple[str | None, str | None]:
        options = parse_mcq_options(question)
        expected = ("pa", "ap") if self.task == "projection" else (
            ("good", "poor") if self.task == "inspiration" else ("yes", "no")
        )
        positive = next(
            (letter for letter, text in options.items() if _normalize(text).startswith(expected[0])),
            None,
        )
        negative = next(
            (letter for letter, text in options.items() if _normalize(text).startswith(expected[1])),
            None,
        )
        return positive, negative

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        if turn.scorer_stage != "final":
            return GateResult(
                passed=False,
                reason="practical_final_verifier_used_for_non_final_stage",
                metadata={"verifier": self.name},
            )
        positive_letter, negative_letter = self._final_polarity(turn.question)
        valid_final_letters = {
            letter for letter in (positive_letter, negative_letter) if letter is not None
        }
        selected_final = selected_option_letters(turn.question, response)
        if (
            len(valid_final_letters) != 2
            or len(selected_final) != 1
            or selected_final[0] not in valid_final_letters
        ):
            return GateResult(
                passed=False,
                score=0.0,
                reason="final_selection_not_single_valid_option",
                metadata={
                    "verifier": self.name,
                    "decision": "REPAIR",
                    "selected_final": list(selected_final),
                },
            )

        measurement_response = accepted_responses.get("measurement")
        if measurement_response is None:
            return GateResult(
                passed=True,
                reason=None,
                metadata={
                    "verifier": self.name,
                    "decision": "ABSTAIN",
                    "abstain_reason": "missing_accepted_measurement",
                },
            )
        inference = self.infer(measurement_response)
        if len(inference.possible_positive) != 1:
            return GateResult(
                passed=True,
                reason=None,
                metadata={
                    "verifier": self.name,
                    "decision": "ABSTAIN",
                    "abstain_reason": inference.reason,
                    "possible_positive": sorted(inference.possible_positive),
                    "evidence": inference.evidence,
                },
            )

        positive = next(iter(inference.possible_positive))
        expected_letter = positive_letter if positive else negative_letter
        passed = selected_final == (expected_letter,)
        return GateResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=None if passed else "final_decision_inconsistent_with_measurement",
            metadata={
                "verifier": self.name,
                "decision": "PASS" if passed else "REPAIR",
                "selected_final": list(selected_final),
                "expected_final": expected_letter,
                "possible_positive": [positive],
                "inference_reason": inference.reason,
                "evidence": inference.evidence,
            },
        )


@dataclass(frozen=True)
class InclusionGeometry:
    margins: dict[str, float]
    mask_fraction: float
    image_size: tuple[int, int]


def inclusion_geometry_from_mask(red_mask: "np.ndarray") -> InclusionGeometry | None:
    """Extract the exact Stage-3 margin feature from a binary lung mask."""

    import numpy as np

    red_mask = np.asarray(red_mask, dtype=bool)
    if red_mask.ndim != 2:
        return None
    height, width = red_mask.shape
    if height < 32 or width < 32:
        return None
    mask_fraction = float(red_mask.mean())
    if mask_fraction < 0.04:
        return None

    midpoint = width // 2
    margins: dict[str, float] = {}
    for patient_side, image_slice, offset in (
        ("right", slice(0, midpoint), 0),
        ("left", slice(midpoint, width), midpoint),
    ):
        y_values, x_values = np.where(red_mask[:, image_slice])
        if len(x_values) < 0.015 * height * width:
            return None
        x_values = x_values + offset
        vertical_span = (int(y_values.max()) - int(y_values.min()) + 1) / height
        horizontal_span = (int(x_values.max()) - int(x_values.min()) + 1) / width
        if vertical_span < 0.35 or horizontal_span < 0.12:
            return None
        margins[f"apex_{patient_side}"] = float(y_values.min() / height)
        margins[f"bottom_{patient_side}"] = float(
            (height - 1 - y_values.max()) / height
        )
        margins[f"side_{patient_side}"] = float(
            x_values.min() / width
            if patient_side == "right"
            else (width - 1 - x_values.max()) / width
        )
    return InclusionGeometry(
        margins=margins,
        mask_fraction=mask_fraction,
        image_size=(width, height),
    )


def extract_inclusion_geometry(image_path: Path) -> InclusionGeometry | None:
    """Extract lung-to-frame margins from a displayed red mask overlay.

    This deliberately uses only pixels that were available to the model at
    Stage 2. It does not inspect the file or directory name.
    """

    import numpy as np
    from PIL import Image

    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    height, width, _ = image.shape
    if height < 32 or width < 32:
        return None
    values = image.astype(np.int16)
    red_mask = (
        (values[:, :, 0] - values[:, :, 1] > 25)
        & (values[:, :, 0] - values[:, :, 2] > 20)
        & (values[:, :, 0] > 80)
    )
    return inclusion_geometry_from_mask(red_mask)


class PracticalInclusionBodypartVerifier:
    """Identify the bilateral lung overlay among Stage-2 image candidates.

    The gate examines only pixels displayed in the current turn.  A repair is
    requested only when exactly one candidate has the bilateral red-mask
    geometry expected for the lung overlay.  Zero or multiple detections are
    inconclusive and therefore abstain as pass.
    """

    name = "practical_inclusion_bodypart_geometry_v0"

    def __init__(self, *, min_mask_fraction: float = 0.21) -> None:
        if not 0.0 < min_mask_fraction < 1.0:
            raise ValueError("min_mask_fraction must be between 0 and 1")
        self.min_mask_fraction = min_mask_fraction

    @staticmethod
    def _special_option_letters(question: str) -> frozenset[str]:
        return frozenset(
            letter
            for letter, option in parse_mcq_options(question).items()
            if "need new option" in _normalize(option)
            or "none of the above" in _normalize(option)
            or _normalize(option) == "none"
        )

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        del accepted_responses
        if turn.scorer_stage != "bodypart":
            return GateResult(
                passed=False,
                reason="inclusion_bodypart_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )

        options = parse_mcq_options(turn.question)
        selected = selected_option_letters(turn.question, response)
        valid_letters = frozenset(options)
        special_letters = self._special_option_letters(turn.question)
        if (
            len(selected) != 1
            or selected[0] not in valid_letters
            or (
                selected[0] not in special_letters
                and ord(selected[0]) - ord("a") >= len(turn.image_paths)
            )
        ):
            return GateResult(
                passed=False,
                score=0.0,
                reason="inclusion_bodypart_not_single_valid_option",
                metadata={
                    "verifier": self.name,
                    "decision": "REPAIR",
                    "selected": list(selected),
                },
            )

        geometries = tuple(
            extract_inclusion_geometry(image_path) for image_path in turn.image_paths
        )
        detected_letters = tuple(
            chr(ord("a") + index)
            for index, geometry in enumerate(geometries)
            if geometry is not None
            and geometry.mask_fraction >= self.min_mask_fraction
        )
        metadata = {
            "verifier": self.name,
            "selected": list(selected),
            "detected_candidate_letters": list(detected_letters),
            "detected_candidate_count": len(detected_letters),
            "min_mask_fraction": self.min_mask_fraction,
        }
        if len(detected_letters) != 1:
            metadata.update(
                {
                    "decision": "ABSTAIN",
                    "abstain_reason": (
                        "no_bilateral_lung_overlay_detected"
                        if not detected_letters
                        else "multiple_bilateral_lung_overlays_detected"
                    ),
                }
            )
            return GateResult(passed=True, metadata=metadata)

        expected = detected_letters[0]
        passed = selected == (expected,)
        metadata.update(
            {
                "decision": "PASS" if passed else "REPAIR",
                "expected_candidate": expected,
            }
        )
        return GateResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=None if passed else "selected_image_is_not_detected_lung_overlay",
            metadata=metadata,
        )


_INCLUSION_LABEL_PATTERNS = {
    "apex_right": r"right\s+apex\s*:\s*(included|excluded)",
    "apex_left": r"left\s+apex\s*:\s*(included|excluded)",
    "side_right": r"right\s+rib\s+edge\s*:\s*(included|excluded)",
    "side_left": r"left\s+rib\s+edge\s*:\s*(included|excluded)",
    "bottom_right": r"right\s+costophrenic\s+angle\s*:\s*(included|excluded)",
    "bottom_left": r"left\s+costophrenic\s+angle\s*:\s*(included|excluded)",
}


def parse_inclusion_labels(option_text: str) -> dict[str, bool] | None:
    labels: dict[str, bool] = {}
    for region, pattern in _INCLUSION_LABEL_PATTERNS.items():
        match = re.search(pattern, option_text, re.IGNORECASE)
        if match is None:
            return None
        labels[region] = match.group(1).lower() == "included"
    return labels


class PracticalInclusionMeasurementVerifier:
    """One-sided Stage-3 inclusion verifier based on accepted lung-mask pixels.

    A region is certified as visible only when the displayed lung mask is at
    least ``min_visible_margin`` away from the image frame. A response that
    calls a certified region excluded is rejected. Near-frame regions remain
    unknown, preventing the heuristic from asserting that they are cropped.

    This is a prototype geometry gate. Its margin must be calibrated on a
    patient-disjoint development set before it can be used in a final study.
    """

    name = "practical_inclusion_geometry_v0"

    def __init__(
        self,
        *,
        anatomy_turns: tuple[NativeStageTurn, ...],
        measurement_question: str,
        min_visible_margin: float = 0.10,
        min_visible_margins: dict[str, float] | None = None,
    ) -> None:
        if not 0.0 < min_visible_margin < 0.5:
            raise ValueError("min_visible_margin must be between 0 and 0.5")
        region_keys = frozenset(_INCLUSION_LABEL_PATTERNS)
        if min_visible_margins is None:
            thresholds = {region: min_visible_margin for region in region_keys}
        else:
            if set(min_visible_margins) != region_keys:
                raise ValueError(
                    "min_visible_margins must contain all and only inclusion regions"
                )
            thresholds = {
                region: float(min_visible_margins[region]) for region in region_keys
            }
            if any(not 0.0 < value < 0.5 for value in thresholds.values()):
                raise ValueError(
                    "All min_visible_margins values must be between 0 and 0.5"
                )
        self.anatomy_turns = anatomy_turns
        self.measurement_question = measurement_question
        self.min_visible_margin = min_visible_margin
        self.min_visible_margins = thresholds

    @classmethod
    def from_case(
        cls,
        case: NativePath1Case,
        *,
        min_visible_margin: float = 0.10,
        min_visible_margins: dict[str, float] | None = None,
    ) -> "PracticalInclusionMeasurementVerifier":
        if case.task != "inclusion":
            raise ValueError("The inclusion verifier only supports the inclusion task")
        return cls(
            anatomy_turns=case.anatomy,
            measurement_question=case.measurement.question,
            min_visible_margin=min_visible_margin,
            min_visible_margins=min_visible_margins,
        )

    def _selected_evidence_paths(
        self, accepted_responses: dict[str, str]
    ) -> tuple[Path, ...]:
        selected: list[Path] = []
        for turn in self.anatomy_turns:
            response = accepted_responses.get(turn.key)
            if response is None:
                continue
            for letter in selected_option_letters(turn.question, response):
                index = ord(letter) - ord("a")
                if 0 <= index < len(turn.image_paths):
                    selected.append(turn.image_paths[index])
        return tuple(dict.fromkeys(selected))

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        if turn.scorer_stage != "measurement":
            return GateResult(
                passed=False,
                reason="inclusion_measurement_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        selected = selected_option_letters(self.measurement_question, response)
        options = parse_mcq_options(self.measurement_question)
        if len(selected) != 1 or selected[0] not in options:
            return GateResult(
                passed=False,
                score=0.0,
                reason="inclusion_measurement_not_single_valid_option",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        labels = parse_inclusion_labels(options[selected[0]])
        if labels is None:
            return GateResult(
                passed=False,
                score=0.0,
                reason="inclusion_measurement_labels_not_parseable",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )

        paths = self._selected_evidence_paths(accepted_responses)
        if len(paths) != 1:
            return GateResult(
                passed=True,
                metadata={
                    "verifier": self.name,
                    "decision": "ABSTAIN",
                    "abstain_reason": "requires_one_accepted_evidence_image",
                    "selected_evidence_images": len(paths),
                },
            )
        geometry = extract_inclusion_geometry(paths[0])
        if geometry is None:
            return GateResult(
                passed=True,
                metadata={
                    "verifier": self.name,
                    "decision": "ABSTAIN",
                    "abstain_reason": "accepted_image_not_bilateral_lung_overlay",
                },
            )

        certified_included = sorted(
            region
            for region, margin in geometry.margins.items()
            if margin >= self.min_visible_margins[region]
        )
        contradictions = sorted(
            region for region in certified_included if labels[region] is False
        )
        metadata = {
            "verifier": self.name,
            "decision": "REPAIR" if contradictions else (
                "PASS" if certified_included else "ABSTAIN"
            ),
            "selected_measurement": selected[0],
            "certified_included": certified_included,
            "contradictions": contradictions,
            "min_visible_margin": self.min_visible_margin,
            "min_visible_margins": self.min_visible_margins,
            "geometry": {
                "margins": geometry.margins,
                "mask_fraction": geometry.mask_fraction,
                "image_size": list(geometry.image_size),
            },
        }
        if contradictions:
            return GateResult(
                passed=False,
                score=0.0,
                reason="reported_exclusion_contradicts_lung_mask_margin",
                metadata=metadata,
            )
        if not certified_included:
            metadata["abstain_reason"] = "no_region_clear_of_frame_margin"
        return GateResult(passed=True, score=1.0, metadata=metadata)

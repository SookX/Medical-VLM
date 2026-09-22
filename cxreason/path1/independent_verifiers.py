"""Practical verifiers backed by independently generated anatomical masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from cxreason.data.cxreasonbench import NativeStageTurn
from cxreason.gates.base import GateResult
from cxreason.path1.practical_verifiers import (
    _normalize,
    _numeric_interval,
    InclusionGeometry,
    inclusion_geometry_from_mask,
    parse_inclusion_labels,
    parse_mcq_options,
    selected_option_letters,
)
from cxreason.vision.chestx_det import ChestXDetPrediction


def extract_red_overlay_mask(image_path: Path) -> np.ndarray | None:
    """Return the displayed red overlay as a binary mask, if one is present."""

    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    values = image.astype(np.int16)
    mask = (
        (values[:, :, 0] - values[:, :, 1] > 25)
        & (values[:, :, 0] - values[:, :, 2] > 20)
        & (values[:, :, 0] > 80)
    )
    return mask if mask.any() else None


def dice_score(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("Dice masks must have the same shape")
    denominator = int(first.sum()) + int(second.sum())
    return 1.0 if denominator == 0 else 2.0 * int((first & second).sum()) / denominator


def iou_score(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("IoU masks must have the same shape")
    union = int((first | second).sum())
    return 1.0 if union == 0 else int((first & second).sum()) / union


@dataclass(frozen=True)
class CandidateOverlap:
    letter: str
    dice: float
    iou: float
    overlay_fraction: float


@dataclass(frozen=True)
class ProjectionCandidateOverlap:
    letter: str
    lung_dice: float
    scapula_dice: float
    overlay_fraction: float


@dataclass(frozen=True)
class CardiothoracicGeometry:
    """Horizontal mask extents used for an independent CTR estimate."""

    heart_width: float
    lung_width: float
    raw_ctr: float


@dataclass(frozen=True)
class ProjectionGeometry:
    """Independent side-specific scapula/lung overlap ratios."""

    right_scapular_area: int
    right_overlap_area: int
    right_ratio: float
    left_scapular_area: int
    left_overlap_area: int
    left_ratio: float


def score_lung_candidates(
    turn: NativeStageTurn, prediction: ChestXDetPrediction
) -> tuple[CandidateOverlap, ...]:
    scores: list[CandidateOverlap] = []
    for index, image_path in enumerate(turn.image_paths):
        overlay = extract_red_overlay_mask(image_path)
        if overlay is None:
            scores.append(CandidateOverlap(chr(ord("a") + index), 0.0, 0.0, 0.0))
            continue
        height, width = overlay.shape
        independent = prediction.project_mask(
            ("Left Lung", "Right Lung"), (width, height)
        )
        scores.append(
            CandidateOverlap(
                letter=chr(ord("a") + index),
                dice=dice_score(overlay, independent),
                iou=iou_score(overlay, independent),
                overlay_fraction=float(overlay.mean()),
            )
        )
    return tuple(scores)


def score_projection_candidates(
    turn: NativeStageTurn, prediction: ChestXDetPrediction
) -> tuple[ProjectionCandidateOverlap, ...]:
    """Score displayed masks against hidden bilateral lung and scapula masks."""

    required = {"Left Lung", "Right Lung", "Left Scapula", "Right Scapula"}
    if not required.issubset(prediction.masks):
        raise ValueError("Projection candidate scoring requires lung and scapula masks")
    scores: list[ProjectionCandidateOverlap] = []
    for index, image_path in enumerate(turn.image_paths):
        letter = chr(ord("a") + index)
        overlay = extract_red_overlay_mask(image_path)
        if overlay is None:
            scores.append(ProjectionCandidateOverlap(letter, 0.0, 0.0, 0.0))
            continue
        height, width = overlay.shape
        lungs = prediction.project_mask(
            ("Left Lung", "Right Lung"), (width, height)
        )
        scapulae = prediction.project_mask(
            ("Left Scapula", "Right Scapula"), (width, height)
        )
        scores.append(
            ProjectionCandidateOverlap(
                letter=letter,
                lung_dice=dice_score(overlay, lungs),
                scapula_dice=dice_score(overlay, scapulae),
                overlay_fraction=float(overlay.mean()),
            )
        )
    return tuple(scores)


def independent_inclusion_geometry(
    prediction: ChestXDetPrediction,
) -> InclusionGeometry | None:
    """Measure lung margins without certifying model-cropped frame boundaries.

    ChestX-Det consumes a centered square crop.  On a portrait radiograph the
    top and bottom of the original frame were never observed; on a landscape
    image the lateral edges were never observed.  Projection padding must not
    be mistaken for anatomical clearance, so affected region margins are set
    to zero (unknown for the one-sided verifier).
    """

    mask = prediction.project_mask(
        ("Left Lung", "Right Lung"), prediction.original_size
    )
    geometry = inclusion_geometry_from_mask(mask)
    if geometry is None:
        return None
    width, height = prediction.original_size
    left, top, right, bottom = prediction.crop_box
    margins = dict(geometry.margins)
    if top > 0:
        margins["apex_right"] = 0.0
        margins["apex_left"] = 0.0
    if bottom < height:
        margins["bottom_right"] = 0.0
        margins["bottom_left"] = 0.0
    if left > 0:
        margins["side_right"] = 0.0
    if right < width:
        margins["side_left"] = 0.0
    return InclusionGeometry(
        margins=margins,
        mask_fraction=geometry.mask_fraction,
        image_size=geometry.image_size,
    )


def independent_cardiothoracic_geometry(
    prediction: ChestXDetPrediction,
) -> CardiothoracicGeometry | None:
    """Measure heart/lung horizontal extents in independent model space.

    A horizontal center crop can truncate the anatomy used by CTR, so those
    predictions are treated as unavailable. Vertical cropping is retained:
    both widths are horizontal and measured in the same model coordinate
    system, making their ratio invariant to the common resize.
    """

    required = {"Left Lung", "Right Lung", "Heart"}
    if not required.issubset(prediction.masks):
        return None
    original_width, _ = prediction.original_size
    crop_left, _, crop_right, _ = prediction.crop_box
    if crop_left > 0 or crop_right < original_width:
        return None
    lung = np.logical_or(
        np.asarray(prediction.masks["Left Lung"], dtype=bool),
        np.asarray(prediction.masks["Right Lung"], dtype=bool),
    )
    heart = np.asarray(prediction.masks["Heart"], dtype=bool)

    def width(mask: np.ndarray) -> float | None:
        columns = np.flatnonzero(mask.any(axis=0))
        if columns.size < 2:
            return None
        return float(columns[-1] - columns[0])

    lung_width = width(lung)
    heart_width = width(heart)
    if lung_width is None or heart_width is None or lung_width <= 0:
        return None
    ratio = heart_width / lung_width
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    return CardiothoracicGeometry(
        heart_width=heart_width,
        lung_width=lung_width,
        raw_ctr=float(ratio),
    )


def independent_projection_geometry(
    prediction: ChestXDetPrediction,
) -> ProjectionGeometry | None:
    """Measure scapular overlap divided by scapular area for both sides."""

    required = {
        "Left Lung",
        "Right Lung",
        "Left Scapula",
        "Right Scapula",
    }
    if not required.issubset(prediction.masks):
        return None

    def measure(scapula_name: str, lung_name: str) -> tuple[int, int, float] | None:
        scapula = np.asarray(prediction.masks[scapula_name], dtype=bool)
        lung = np.asarray(prediction.masks[lung_name], dtype=bool)
        area = int(scapula.sum())
        if area <= 0:
            return None
        overlap = int(np.logical_and(scapula, lung).sum())
        return area, overlap, overlap / area

    right = measure("Right Scapula", "Right Lung")
    left = measure("Left Scapula", "Left Lung")
    if right is None or left is None:
        return None
    return ProjectionGeometry(
        right_scapular_area=right[0],
        right_overlap_area=right[1],
        right_ratio=right[2],
        left_scapular_area=left[0],
        left_overlap_area=left[1],
        left_ratio=left[2],
    )


class ProjectionCriterionVerifier:
    """Verify the direct projection criterion from option semantics alone.

    The direct radiographic criterion is scapular retraction versus overlap
    with the lung fields.  If that option is absent, the verifier selects the
    question's explicit continuation/no-match option.  It never receives the
    benchmark answer and abstains if either mapping is ambiguous.
    """

    name = "practical_projection_criterion_semantics_v0"

    @staticmethod
    def _is_direct_criterion(option: str) -> bool:
        normalized = _normalize(option)
        has_scapula = "scapula" in normalized or "scapulae" in normalized
        has_position = "retract" in normalized or "overlap" in normalized
        has_lung = "lung field" in normalized
        return has_scapula and has_position and has_lung

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        del accepted_responses
        if turn.scorer_stage != "criteria":
            return GateResult(
                passed=False,
                reason="projection_criterion_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        options = parse_mcq_options(turn.question)
        direct = [
            letter for letter, option in options.items() if self._is_direct_criterion(option)
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
            expected = direct[0]
            mapping = "direct_scapular_criterion"
        elif not direct and len(continuation) == 1:
            expected = continuation[0]
            mapping = "request_new_options"
        elif not direct and not continuation and len(no_match) == 1:
            expected = no_match[0]
            mapping = "none_of_the_above"
        else:
            return GateResult(
                passed=True,
                metadata={
                    "verifier": self.name,
                    "decision": "ABSTAIN",
                    "abstain_reason": "ambiguous_projection_criterion_mapping",
                    "direct_options": direct,
                    "continuation_options": continuation,
                    "no_match_options": no_match,
                },
            )
        selected = selected_option_letters(turn.question, response)
        passed = selected == (expected,)
        return GateResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=None if passed else "selection_disagrees_with_projection_criterion",
            metadata={
                "verifier": self.name,
                "decision": "PASS" if passed else "REPAIR",
                "mapping": mapping,
                "expected_option": expected,
                "selected": list(selected),
            },
        )


class RefinedCriterionApplicabilityVerifier:
    """Require an affirmative answer to an explicitly applicable refinement.

    This gate reads only the question and model response.  It does not receive
    the benchmark answer, and it abstains on any prompt outside the narrow
    yes/no applicability protocol.
    """

    name = "practical_refined_criterion_applicability_v0"

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        del accepted_responses
        if turn.scorer_stage != "custom_criteria":
            return GateResult(
                passed=False,
                reason="applicability_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        normalized_question = _normalize(turn.question)
        options = parse_mcq_options(turn.question)
        yes = [letter for letter, value in options.items() if _normalize(value) == "yes"]
        no = [
            letter
            for letter, value in options.items()
            if _normalize(value) == "no" or _normalize(value).startswith("no ")
        ]
        protocol_matches = (
            len(yes) == 1
            and len(no) == 1
            and "can you apply this refined criterion" in normalized_question
            and "overlap ratio for each side" in normalized_question
        )
        if not protocol_matches:
            return GateResult(
                passed=True,
                metadata={
                    "verifier": self.name,
                    "decision": "ABSTAIN",
                    "abstain_reason": "unrecognized_applicability_protocol",
                },
            )
        selected = selected_option_letters(turn.question, response)
        passed = selected == (yes[0],)
        return GateResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=None if passed else "refined_criterion_not_accepted",
            metadata={
                "verifier": self.name,
                "decision": "PASS" if passed else "REPAIR",
                "selected": list(selected),
                "affirmative_option": yes[0],
            },
        )


class IndependentProjectionBodypartVerifier:
    """Select lung/scapula candidates using an independent ChestX-Det mask.

    All thresholds are mandatory external-calibration outputs.  A candidate in
    an uncertainty band, a duplicate anatomy match, or an incomplete ordinary
    candidate set makes the verifier abstain instead of risking a false repair.
    """

    name = "independent_chestx_det_projection_bodypart_v0"

    def __init__(
        self,
        *,
        prediction: ChestXDetPrediction,
        min_dice: dict[str, float],
        max_negative_dice: dict[str, float],
        min_class_margin: dict[str, float],
    ) -> None:
        expected = {"lung", "scapula"}
        if set(min_dice) != expected or set(max_negative_dice) != expected:
            raise ValueError("Projection Dice policies require lung and scapula")
        if set(min_class_margin) != expected:
            raise ValueError("Projection margin policies require lung and scapula")
        self.min_dice = {key: float(value) for key, value in min_dice.items()}
        self.max_negative_dice = {
            key: float(value) for key, value in max_negative_dice.items()
        }
        self.min_class_margin = {
            key: float(value) for key, value in min_class_margin.items()
        }
        for anatomy in expected:
            if not 0.0 < self.min_dice[anatomy] < 1.0:
                raise ValueError("Projection minimum Dice values must lie in (0, 1)")
            if not 0.0 <= self.max_negative_dice[anatomy] < self.min_dice[anatomy]:
                raise ValueError("Negative Dice limit must be below minimum Dice")
            if not 0.0 <= self.min_class_margin[anatomy] < 1.0:
                raise ValueError("Projection class margins must lie in [0, 1)")
        self.prediction = prediction

    @staticmethod
    def _special_letters(question: str) -> dict[str, str]:
        special: dict[str, str] = {}
        for letter, option in parse_mcq_options(question).items():
            normalized = _normalize(option)
            if "need new option" in normalized:
                special["need_new"] = letter
            elif "none of the above" in normalized or normalized == "none":
                special["none"] = letter
        return special

    def _classify(self, score: ProjectionCandidateOverlap) -> str | None:
        values = {"lung": score.lung_dice, "scapula": score.scapula_dice}
        eligible = [
            anatomy
            for anatomy, value in values.items()
            if value >= self.min_dice[anatomy]
            and value - values["scapula" if anatomy == "lung" else "lung"]
            >= self.min_class_margin[anatomy]
        ]
        if len(eligible) == 1:
            return eligible[0]
        if all(
            values[anatomy] <= self.max_negative_dice[anatomy]
            for anatomy in ("lung", "scapula")
        ):
            return "negative"
        return None

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
                reason="independent_projection_bodypart_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        options = parse_mcq_options(turn.question)
        selected = selected_option_letters(turn.question, response)
        if not selected or any(letter not in options for letter in selected):
            return GateResult(
                passed=False,
                score=0.0,
                reason="projection_bodypart_has_no_valid_selection",
                metadata={
                    "verifier": self.name,
                    "decision": "REPAIR",
                    "selected": list(selected),
                },
            )
        scores = score_projection_candidates(turn, self.prediction)
        classifications = {score.letter: self._classify(score) for score in scores}
        metadata = {
            "verifier": self.name,
            "selected": list(selected),
            "candidate_scores": [
                {
                    "letter": score.letter,
                    "lung_dice": score.lung_dice,
                    "scapula_dice": score.scapula_dice,
                    "overlay_fraction": score.overlay_fraction,
                    "classification": classifications[score.letter],
                }
                for score in scores
            ],
            "policy": {
                "min_dice": self.min_dice,
                "max_negative_dice": self.max_negative_dice,
                "min_class_margin": self.min_class_margin,
            },
        }
        if any(value is None for value in classifications.values()):
            metadata.update(
                {"decision": "ABSTAIN", "abstain_reason": "uncertain_candidate"}
            )
            return GateResult(passed=True, metadata=metadata)
        positive = {
            anatomy: sorted(
                letter for letter, value in classifications.items() if value == anatomy
            )
            for anatomy in ("lung", "scapula")
        }
        if any(len(letters) > 1 for letters in positive.values()):
            metadata.update(
                {"decision": "ABSTAIN", "abstain_reason": "duplicate_anatomy_match"}
            )
            return GateResult(passed=True, metadata=metadata)
        relevant = sorted(
            [letter for letters in positive.values() for letter in letters]
        )
        special = self._special_letters(turn.question)
        if "need_new" in special:
            expected = relevant + (
                [special["need_new"]]
                if any(not positive[anatomy] for anatomy in ("lung", "scapula"))
                else []
            )
        elif "none" in special:
            expected = relevant if relevant else [special["none"]]
        else:
            expected = relevant
        if not expected:
            metadata.update(
                {"decision": "ABSTAIN", "abstain_reason": "no_expected_selection"}
            )
            return GateResult(passed=True, metadata=metadata)
        passed = set(selected) == set(expected) and len(selected) == len(expected)
        metadata.update(
            {
                "decision": "PASS" if passed else "REPAIR",
                "expected_candidates": expected,
                "certified_anatomy": positive,
            }
        )
        return GateResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=None if passed else "selection_disagrees_with_independent_anatomy",
            metadata=metadata,
        )


class IndependentInclusionBodypartVerifier:
    """Match displayed Stage-2 candidates to an independent bilateral-lung mask.

    Thresholds are mandatory because they must eventually come from an external
    development set. This class intentionally supplies no benchmark-tuned
    defaults.
    """

    name = "independent_chestx_det_inclusion_bodypart_v0"

    def __init__(
        self,
        *,
        prediction: ChestXDetPrediction,
        min_dice: float,
        max_special_dice: float,
        min_dice_margin: float,
    ) -> None:
        if not 0.0 < min_dice < 1.0:
            raise ValueError("min_dice must be between 0 and 1")
        if not 0.0 <= min_dice_margin < 1.0:
            raise ValueError("min_dice_margin must be in [0, 1)")
        if not 0.0 <= max_special_dice < min_dice:
            raise ValueError("max_special_dice must be in [0, min_dice)")
        self.prediction = prediction
        self.min_dice = min_dice
        self.max_special_dice = max_special_dice
        self.min_dice_margin = min_dice_margin

    @staticmethod
    def _special_letters(question: str) -> tuple[str, ...]:
        return tuple(
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
                reason="independent_inclusion_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        selected = selected_option_letters(turn.question, response)
        options = parse_mcq_options(turn.question)
        if len(selected) != 1 or selected[0] not in options:
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
        scores = score_lung_candidates(turn, self.prediction)
        ranked = sorted(scores, key=lambda value: (-value.dice, value.letter))
        special = self._special_letters(turn.question)
        metadata = {
            "verifier": self.name,
            "selected": list(selected),
            "min_dice": self.min_dice,
            "max_special_dice": self.max_special_dice,
            "min_dice_margin": self.min_dice_margin,
            "candidate_scores": [
                {
                    "letter": score.letter,
                    "dice": score.dice,
                    "iou": score.iou,
                    "overlay_fraction": score.overlay_fraction,
                }
                for score in scores
            ],
        }
        if not ranked or ranked[0].dice <= self.max_special_dice:
            if len(special) == 1:
                expected = special[0]
            else:
                metadata.update(
                    {"decision": "ABSTAIN", "abstain_reason": "no_candidate_match"}
                )
                return GateResult(passed=True, metadata=metadata)
        elif ranked[0].dice < self.min_dice:
            metadata.update(
                {"decision": "ABSTAIN", "abstain_reason": "uncertain_candidate_band"}
            )
            return GateResult(passed=True, metadata=metadata)
        elif len(ranked) > 1 and ranked[0].dice - ranked[1].dice < self.min_dice_margin:
            metadata.update(
                {"decision": "ABSTAIN", "abstain_reason": "ambiguous_candidate_match"}
            )
            return GateResult(passed=True, metadata=metadata)
        else:
            expected = ranked[0].letter
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
            reason=None if passed else "selected_image_disagrees_with_independent_mask",
            metadata=metadata,
        )


class IndependentInclusionMeasurementVerifier:
    """One-sided Stage-3 verifier using a hidden independent lung mask."""

    name = "independent_chestx_det_inclusion_measurement_v0"

    def __init__(
        self,
        *,
        prediction: ChestXDetPrediction,
        measurement_question: str,
        min_visible_margins: dict[str, float],
    ) -> None:
        expected = {
            "apex_right",
            "apex_left",
            "side_right",
            "side_left",
            "bottom_right",
            "bottom_left",
        }
        if set(min_visible_margins) != expected:
            raise ValueError("min_visible_margins must contain all six inclusion regions")
        thresholds = {key: float(value) for key, value in min_visible_margins.items()}
        if any(not 0.0 < value < 0.5 for value in thresholds.values()):
            raise ValueError("All Stage-3 margins must lie in (0, 0.5)")
        self.prediction = prediction
        self.measurement_question = measurement_question
        self.min_visible_margins = thresholds

    def geometry(self):
        return independent_inclusion_geometry(self.prediction)

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        del accepted_responses
        if turn.scorer_stage != "measurement":
            return GateResult(
                passed=False,
                reason="independent_inclusion_measurement_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        selected = selected_option_letters(self.measurement_question, response)
        options = parse_mcq_options(self.measurement_question)
        if len(selected) != 1 or selected[0] not in options:
            return GateResult(
                passed=False,
                score=0.0,
                reason="inclusion_measurement_not_single_valid_option",
                metadata={
                    "verifier": self.name,
                    "decision": "REPAIR",
                    "selected": list(selected),
                },
            )
        labels = parse_inclusion_labels(options[selected[0]])
        if labels is None:
            return GateResult(
                passed=False,
                score=0.0,
                reason="inclusion_measurement_labels_not_parseable",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        geometry = self.geometry()
        if geometry is None:
            return GateResult(
                passed=True,
                metadata={
                    "verifier": self.name,
                    "decision": "ABSTAIN",
                    "abstain_reason": "independent_lung_geometry_unavailable",
                },
            )
        certified = sorted(
            region
            for region, margin in geometry.margins.items()
            if margin >= self.min_visible_margins[region]
        )
        contradictions = sorted(region for region in certified if labels[region] is False)
        decision = "REPAIR" if contradictions else ("PASS" if certified else "ABSTAIN")
        metadata = {
            "verifier": self.name,
            "decision": decision,
            "selected_measurement": selected[0],
            "certified_included": certified,
            "contradictions": contradictions,
            "min_visible_margins": self.min_visible_margins,
            "geometry": {
                "margins": geometry.margins,
                "mask_fraction": geometry.mask_fraction,
                "image_size": list(geometry.image_size),
            },
        }
        return GateResult(
            passed=not contradictions,
            score=0.0 if contradictions else 1.0,
            reason=(
                "measurement_contradicts_independent_lung_geometry"
                if contradictions
                else None
            ),
            metadata=metadata,
        )


class IndependentCardiothoracicMeasurementVerifier:
    """Reject CTR options disjoint from an externally calibrated interval."""

    name = "independent_chestx_det_cardiothoracic_measurement_v0"

    def __init__(
        self,
        *,
        prediction: ChestXDetPrediction,
        measurement_question: str,
        slope: float,
        intercept: float,
        residual_low: float,
        residual_high: float,
    ) -> None:
        values = (slope, intercept, residual_low, residual_high)
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("CTR calibration values must be finite")
        if slope <= 0:
            raise ValueError("CTR calibration slope must be positive")
        if residual_low > residual_high:
            raise ValueError("CTR residual interval is reversed")
        self.prediction = prediction
        self.measurement_question = measurement_question
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.residual_low = float(residual_low)
        self.residual_high = float(residual_high)

    def geometry(self) -> CardiothoracicGeometry | None:
        return independent_cardiothoracic_geometry(self.prediction)

    def plausible_interval(
        self, geometry: CardiothoracicGeometry
    ) -> tuple[float, float]:
        estimate = self.intercept + self.slope * geometry.raw_ctr
        return (
            max(0.0, estimate + self.residual_low),
            min(1.0, estimate + self.residual_high),
        )

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        del accepted_responses
        if turn.scorer_stage != "measurement":
            return GateResult(
                passed=False,
                reason="independent_ctr_measurement_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        options = parse_mcq_options(self.measurement_question)
        selected = selected_option_letters(self.measurement_question, response)
        if len(selected) != 1 or selected[0] not in options:
            return GateResult(
                passed=False,
                score=0.0,
                reason="ctr_measurement_not_single_valid_option",
                metadata={
                    "verifier": self.name,
                    "decision": "REPAIR",
                    "selected": list(selected),
                },
            )
        selected_interval = _numeric_interval(options[selected[0]])
        if selected_interval is None:
            return GateResult(
                passed=False,
                score=0.0,
                reason="ctr_measurement_interval_not_parseable",
                metadata={
                    "verifier": self.name,
                    "decision": "REPAIR",
                    "selected": selected[0],
                },
            )
        geometry = self.geometry()
        if geometry is None:
            return GateResult(
                passed=True,
                metadata={
                    "verifier": self.name,
                    "decision": "ABSTAIN",
                    "abstain_reason": "independent_ctr_geometry_unavailable",
                },
            )
        plausible = self.plausible_interval(geometry)
        compatible = max(plausible[0], selected_interval[0]) <= min(
            plausible[1], selected_interval[1]
        ) + 1e-12
        estimate = self.intercept + self.slope * geometry.raw_ctr
        metadata = {
            "verifier": self.name,
            "decision": "PASS" if compatible else "REPAIR",
            "selected_measurement": selected[0],
            "selected_interval": list(selected_interval),
            "plausible_ctr_interval": list(plausible),
            "calibrated_ctr_estimate": estimate,
            "raw_ctr": geometry.raw_ctr,
            "heart_width": geometry.heart_width,
            "lung_width": geometry.lung_width,
            "calibration": {
                "slope": self.slope,
                "intercept": self.intercept,
                "residual_low": self.residual_low,
                "residual_high": self.residual_high,
            },
        }
        return GateResult(
            passed=compatible,
            score=1.0 if compatible else 0.0,
            reason=None if compatible else "selected_ctr_disjoint_from_independent_interval",
            metadata=metadata,
        )


class IndependentProjectionMeasurementVerifier:
    """Verify paired projection ranges using independent scapula/lung masks."""

    name = "independent_chestx_det_projection_measurement_v0"

    def __init__(
        self,
        *,
        prediction: ChestXDetPrediction,
        measurement_question: str,
        side_policies: dict[str, dict[str, float]],
    ) -> None:
        if set(side_policies) != {"right", "left"}:
            raise ValueError("Projection policy must contain right and left sides")
        self.side_policies: dict[str, dict[str, float]] = {}
        for side, policy in side_policies.items():
            if set(policy) != {"intercept", "slope", "residual_low", "residual_high"}:
                raise ValueError(f"Incomplete projection policy for {side}")
            values = {key: float(value) for key, value in policy.items()}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("Projection calibration values must be finite")
            if values["slope"] <= 0 or values["residual_low"] > values["residual_high"]:
                raise ValueError(f"Invalid projection calibration for {side}")
            self.side_policies[side] = values
        self.prediction = prediction
        self.measurement_question = measurement_question

    def geometry(self) -> ProjectionGeometry | None:
        return independent_projection_geometry(self.prediction)

    def plausible_interval(self, side: str, raw_ratio: float) -> tuple[float, float]:
        policy = self.side_policies[side]
        estimate = policy["intercept"] + policy["slope"] * raw_ratio
        return (
            max(0.0, estimate + policy["residual_low"]),
            min(1.0, estimate + policy["residual_high"]),
        )

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        del accepted_responses
        if turn.scorer_stage != "measurement":
            return GateResult(
                passed=False,
                reason="independent_projection_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        options = parse_mcq_options(self.measurement_question)
        selected = selected_option_letters(self.measurement_question, response)
        parsed: dict[str, tuple[str, tuple[float, float]]] = {}
        for letter in selected:
            option = options.get(letter, "")
            normalized = _normalize(option)
            side = "right" if normalized.startswith("right") else (
                "left" if normalized.startswith("left") else None
            )
            interval = _numeric_interval(option)
            if side is not None and interval is not None and side not in parsed:
                parsed[side] = (letter, interval)
        if len(selected) != 2 or set(parsed) != {"right", "left"}:
            return GateResult(
                passed=False,
                score=0.0,
                reason="projection_measurement_requires_one_option_per_side",
                metadata={
                    "verifier": self.name,
                    "decision": "REPAIR",
                    "selected": list(selected),
                },
            )
        geometry = self.geometry()
        if geometry is None:
            return GateResult(
                passed=True,
                metadata={
                    "verifier": self.name,
                    "decision": "ABSTAIN",
                    "abstain_reason": "independent_projection_geometry_unavailable",
                },
            )
        raw = {"right": geometry.right_ratio, "left": geometry.left_ratio}
        plausible = {
            side: self.plausible_interval(side, raw[side]) for side in ("right", "left")
        }
        contradictions = [
            side
            for side in ("right", "left")
            if max(plausible[side][0], parsed[side][1][0])
            > min(plausible[side][1], parsed[side][1][1]) + 1e-12
        ]
        metadata = {
            "verifier": self.name,
            "decision": "REPAIR" if contradictions else "PASS",
            "selected_measurements": {
                side: {"letter": parsed[side][0], "interval": list(parsed[side][1])}
                for side in ("right", "left")
            },
            "plausible_ratio_intervals": {
                side: list(plausible[side]) for side in ("right", "left")
            },
            "raw_ratios": raw,
            "contradictions": contradictions,
            "geometry": {
                "right_scapular_area": geometry.right_scapular_area,
                "right_overlap_area": geometry.right_overlap_area,
                "left_scapular_area": geometry.left_scapular_area,
                "left_overlap_area": geometry.left_overlap_area,
            },
            "calibration": self.side_policies,
        }
        return GateResult(
            passed=not contradictions,
            score=0.0 if contradictions else 1.0,
            reason=(
                "selected_projection_range_disjoint_from_independent_interval"
                if contradictions
                else None
            ),
            metadata=metadata,
        )

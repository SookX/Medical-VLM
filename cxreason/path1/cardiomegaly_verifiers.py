"""Independent experimental verifiers for Cardiomegaly Path 1 stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cxreason.data.cxreasonbench import NativeStageTurn
from cxreason.gates.base import GateResult
from cxreason.path1.independent_verifiers import dice_score, extract_red_overlay_mask
from cxreason.path1.practical_verifiers import (
    _normalize,
    parse_mcq_options,
    selected_option_letters,
)
from cxreason.vision.chestx_det import ChestXDetPrediction


@dataclass(frozen=True)
class CardiomegalyCandidateEvidence:
    letter: str
    heart_dice: float
    thoracic_span_iou: float
    overlay_fraction: float
    bbox_aspect_ratio: float
    max_row_coverage: float = 0.0
    max_column_coverage: float = 0.0
    bbox_fill_fraction: float = 0.0
    bbox_height_fraction: float = 0.0


def cardiomegaly_candidate_evidence_from_masks(
    overlay: np.ndarray,
    heart: np.ndarray,
    lungs: np.ndarray,
    *,
    letter: str = "",
) -> CardiomegalyCandidateEvidence:
    """Score a displayed candidate against independent heart/lung evidence.

    Filled candidates are compared to the independent heart mask. Thin,
    horizontal candidates are compared by their one-dimensional span to the
    independent bilateral-lung span. Width rulers contain a full horizontal
    bar and vertical end-caps, so both axes must contain a nearly complete
    cross-section. This is invariant to the end-cap height that makes bounding
    box aspect ratio unreliable across renderers.
    """

    overlay = np.asarray(overlay, dtype=bool)
    heart = np.asarray(heart, dtype=bool)
    lungs = np.asarray(lungs, dtype=bool)
    if overlay.shape != heart.shape or overlay.shape != lungs.shape:
        raise ValueError("Cardiomegaly evidence masks must share one shape")
    rows, columns = np.where(overlay)
    if not len(columns):
        return CardiomegalyCandidateEvidence(letter, 0.0, 0.0, 0.0, 0.0)
    overlay_xmin, overlay_xmax = int(columns.min()), int(columns.max())
    overlay_ymin, overlay_ymax = int(rows.min()), int(rows.max())
    width = overlay_xmax - overlay_xmin + 1
    height = overlay_ymax - overlay_ymin + 1
    aspect = width / height
    cropped = overlay[overlay_ymin : overlay_ymax + 1, overlay_xmin : overlay_xmax + 1]
    max_row_coverage = float(cropped.sum(axis=1).max() / width)
    max_column_coverage = float(cropped.sum(axis=0).max() / height)
    bbox_fill_fraction = float(cropped.mean())
    bbox_height_fraction = float(height / overlay.shape[0])
    lung_columns = np.flatnonzero(lungs.any(axis=0))
    span_iou = 0.0
    line_representation = (aspect >= 5.0 and bbox_height_fraction <= 0.03) or (
        max_row_coverage >= 0.98
        and max_column_coverage >= 0.98
        and bbox_fill_fraction <= 0.30
    )
    if line_representation and lung_columns.size:
        lung_xmin, lung_xmax = int(lung_columns[0]), int(lung_columns[-1])
        intersection = max(
            0, min(overlay_xmax, lung_xmax) - max(overlay_xmin, lung_xmin) + 1
        )
        union = max(overlay_xmax, lung_xmax) - min(overlay_xmin, lung_xmin) + 1
        span_iou = intersection / union
    return CardiomegalyCandidateEvidence(
        letter=letter,
        heart_dice=dice_score(overlay, heart),
        thoracic_span_iou=float(span_iou),
        overlay_fraction=float(overlay.mean()),
        bbox_aspect_ratio=float(aspect),
        max_row_coverage=max_row_coverage,
        max_column_coverage=max_column_coverage,
        bbox_fill_fraction=bbox_fill_fraction,
        bbox_height_fraction=bbox_height_fraction,
    )


def score_cardiomegaly_candidates(
    turn: NativeStageTurn, prediction: ChestXDetPrediction
) -> tuple[CardiomegalyCandidateEvidence, ...]:
    required = {"Left Lung", "Right Lung", "Heart"}
    if not required.issubset(prediction.masks):
        raise ValueError("Cardiomegaly scoring requires heart and bilateral lungs")
    scores: list[CardiomegalyCandidateEvidence] = []
    for index, image_path in enumerate(turn.image_paths):
        letter = chr(ord("a") + index)
        overlay = extract_red_overlay_mask(image_path)
        if overlay is None:
            scores.append(CardiomegalyCandidateEvidence(letter, 0.0, 0.0, 0.0, 0.0))
            continue
        height, width = overlay.shape
        heart = prediction.project_mask("Heart", (width, height))
        lungs = prediction.project_mask(("Left Lung", "Right Lung"), (width, height))
        scores.append(
            cardiomegaly_candidate_evidence_from_masks(
                overlay, heart, lungs, letter=letter
            )
        )
    return tuple(scores)


class IndependentCardiomegalyBodypartVerifier:
    """Ground Heart and cardiothoracic-width candidates independently."""

    name = "independent_chestx_det_cardiomegaly_bodypart_v0"

    def __init__(
        self,
        *,
        prediction: ChestXDetPrediction,
        min_heart_dice: float,
        min_thoracic_span_iou: float,
        max_negative_heart_dice: float,
        max_negative_thoracic_span_iou: float,
    ) -> None:
        values = (
            min_heart_dice,
            min_thoracic_span_iou,
            max_negative_heart_dice,
            max_negative_thoracic_span_iou,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError("Cardiomegaly Stage-2 thresholds must lie in [0, 1]")
        if max_negative_heart_dice >= min_heart_dice:
            raise ValueError("Negative Heart Dice must be below positive Heart Dice")
        if max_negative_thoracic_span_iou >= min_thoracic_span_iou:
            raise ValueError("Negative thoracic IoU must be below positive thoracic IoU")
        self.prediction = prediction
        self.min_heart_dice = float(min_heart_dice)
        self.min_thoracic_span_iou = float(min_thoracic_span_iou)
        self.max_negative_heart_dice = float(max_negative_heart_dice)
        self.max_negative_thoracic_span_iou = float(
            max_negative_thoracic_span_iou
        )

    def _classify(self, score: CardiomegalyCandidateEvidence) -> str | None:
        heart = score.heart_dice >= self.min_heart_dice
        thoracic = score.thoracic_span_iou >= self.min_thoracic_span_iou
        if heart and not thoracic:
            return "heart"
        if thoracic and not heart:
            return "thoracic_width"
        if (
            score.heart_dice <= self.max_negative_heart_dice
            and score.thoracic_span_iou
            <= self.max_negative_thoracic_span_iou
        ):
            return "negative"
        return None

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
                reason="independent_cardiomegaly_bodypart_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        options = parse_mcq_options(turn.question)
        selected = selected_option_letters(turn.question, response)
        if not selected or any(letter not in options for letter in selected):
            return GateResult(
                passed=False,
                score=0.0,
                reason="cardiomegaly_bodypart_has_no_valid_selection",
                metadata={
                    "verifier": self.name,
                    "decision": "REPAIR",
                    "selected": list(selected),
                },
            )
        scores = score_cardiomegaly_candidates(turn, self.prediction)
        classifications = {score.letter: self._classify(score) for score in scores}
        metadata = {
            "verifier": self.name,
            "selected": list(selected),
            "candidate_scores": [
                {
                    "letter": score.letter,
                    "heart_dice": score.heart_dice,
                    "thoracic_span_iou": score.thoracic_span_iou,
                    "overlay_fraction": score.overlay_fraction,
                    "bbox_aspect_ratio": score.bbox_aspect_ratio,
                    "max_row_coverage": score.max_row_coverage,
                    "max_column_coverage": score.max_column_coverage,
                    "bbox_fill_fraction": score.bbox_fill_fraction,
                    "bbox_height_fraction": score.bbox_height_fraction,
                    "classification": classifications[score.letter],
                }
                for score in scores
            ],
            "policy": {
                "min_heart_dice": self.min_heart_dice,
                "min_thoracic_span_iou": self.min_thoracic_span_iou,
                "max_negative_heart_dice": self.max_negative_heart_dice,
                "max_negative_thoracic_span_iou": (
                    self.max_negative_thoracic_span_iou
                ),
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
            for anatomy in ("heart", "thoracic_width")
        }
        if any(len(letters) > 1 for letters in positive.values()):
            metadata.update(
                {"decision": "ABSTAIN", "abstain_reason": "duplicate_anatomy_match"}
            )
            return GateResult(passed=True, metadata=metadata)
        relevant = sorted(letter for values in positive.values() for letter in values)
        special = self._special_letters(turn.question)
        if "need_new" in special:
            expected = relevant + (
                [special["need_new"]]
                if any(not positive[name] for name in positive)
                else []
            )
        elif relevant:
            expected = relevant
        elif "none" in special:
            expected = [special["none"]]
        else:
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
            reason=(
                None
                if passed
                else "selection_disagrees_with_independent_cardiomegaly_anatomy"
            ),
            metadata=metadata,
        )

"""Independent experimental verifiers for Mediastinal Widening Path 1."""

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


MAX_LINE_BBOX_FILL_RATIO = 0.30
MIN_LINE_RUN_FRACTION = 0.20
LUNG_ROW_BAND_FRACTION = 0.015


@dataclass(frozen=True)
class MediastinalCandidateEvidence:
    letter: str
    mediastinum_dice: float
    thoracic_line_score: float
    overlay_fraction: float
    bbox_fill_ratio: float
    longest_run_fraction: float
    horizontal_row_fraction: float
    line_anchor_row_fraction: float
    line_anchor_alignment: float


def _longest_horizontal_run(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return row, left, right, and length of the longest contiguous run."""

    best = (0, 0, -1, 0)
    for row_index in np.flatnonzero(mask.any(axis=1)):
        row = np.asarray(mask[row_index], dtype=np.int8)
        changes = np.diff(np.pad(row, (1, 1)))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        if not starts.size:
            continue
        lengths = ends - starts + 1
        index = int(np.argmax(lengths))
        candidate = (
            int(row_index),
            int(starts[index]),
            int(ends[index]),
            int(lengths[index]),
        )
        if candidate[3] > best[3]:
            best = candidate
    return best


def mediastinal_candidate_evidence_from_masks(
    overlay: np.ndarray,
    cardiomediastinum: np.ndarray,
    lungs: np.ndarray,
    line_anchor: np.ndarray | None = None,
    *,
    letter: str = "",
) -> MediastinalCandidateEvidence:
    overlay = np.asarray(overlay, dtype=bool)
    cardiomediastinum = np.asarray(cardiomediastinum, dtype=bool)
    lungs = np.asarray(lungs, dtype=bool)
    line_anchor = (
        np.asarray(line_anchor, dtype=bool)
        if line_anchor is not None
        else cardiomediastinum
    )
    if (
        overlay.shape != cardiomediastinum.shape
        or overlay.shape != lungs.shape
        or overlay.shape != line_anchor.shape
    ):
        raise ValueError("Mediastinal evidence masks must share one shape")
    rows, columns = np.where(overlay)
    if not len(columns):
        return MediastinalCandidateEvidence(
            letter, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
    bbox_width = int(columns.max()) - int(columns.min()) + 1
    bbox_height = int(rows.max()) - int(rows.min()) + 1
    bbox_fill = float(overlay.sum()) / (bbox_width * bbox_height)
    line_row, line_left, line_right, line_length = _longest_horizontal_run(overlay)
    line_fraction = line_length / overlay.shape[1]
    line_score = 0.0
    anchor_widths = np.zeros(line_anchor.shape[0], dtype=np.int32)
    for anchor_row in np.flatnonzero(line_anchor.any(axis=1)):
        anchor_columns = np.flatnonzero(line_anchor[anchor_row])
        anchor_widths[anchor_row] = int(anchor_columns[-1] - anchor_columns[0] + 1)
    anchor_row = int(np.argmax(anchor_widths))
    alignment_scale = max(1.0, overlay.shape[0] * 0.25)
    anchor_alignment = max(0.0, 1.0 - abs(line_row - anchor_row) / alignment_scale)
    if (
        bbox_fill <= MAX_LINE_BBOX_FILL_RATIO
        and line_fraction >= MIN_LINE_RUN_FRACTION
    ):
        band = max(1, round(overlay.shape[0] * LUNG_ROW_BAND_FRACTION))
        lung_columns = np.flatnonzero(
            lungs[
                max(0, line_row - band) : min(overlay.shape[0], line_row + band + 1)
            ].any(axis=0)
        )
        if lung_columns.size:
            lung_left, lung_right = int(lung_columns[0]), int(lung_columns[-1])
            intersection = max(
                0, min(line_right, lung_right) - max(line_left, lung_left) + 1
            )
            union = max(line_right, lung_right) - min(line_left, lung_left) + 1
            line_score = (intersection / union) * anchor_alignment
    return MediastinalCandidateEvidence(
        letter=letter,
        mediastinum_dice=dice_score(overlay, cardiomediastinum),
        thoracic_line_score=float(line_score),
        overlay_fraction=float(overlay.mean()),
        bbox_fill_ratio=bbox_fill,
        longest_run_fraction=float(line_fraction),
        horizontal_row_fraction=float(line_row / max(1, overlay.shape[0] - 1)),
        line_anchor_row_fraction=float(anchor_row / max(1, overlay.shape[0] - 1)),
        line_anchor_alignment=float(anchor_alignment),
    )


def score_mediastinal_candidates(
    turn: NativeStageTurn, prediction: ChestXDetPrediction
) -> tuple[MediastinalCandidateEvidence, ...]:
    required = {"Left Lung", "Right Lung", "Heart", "Mediastinum"}
    if not required.issubset(prediction.masks):
        raise ValueError(
            "Mediastinal scoring requires heart, mediastinum, and bilateral lungs"
        )
    scores: list[MediastinalCandidateEvidence] = []
    for index, image_path in enumerate(turn.image_paths):
        letter = chr(ord("a") + index)
        overlay = extract_red_overlay_mask(image_path)
        if overlay is None:
            scores.append(
                MediastinalCandidateEvidence(
                    letter, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                )
            )
            continue
        height, width = overlay.shape
        mediastinum = prediction.project_mask("Mediastinum", (width, height))
        heart = prediction.project_mask("Heart", (width, height))
        lungs = prediction.project_mask(("Left Lung", "Right Lung"), (width, height))
        scores.append(
            mediastinal_candidate_evidence_from_masks(
                overlay,
                mediastinum | heart,
                lungs,
                line_anchor=mediastinum,
                letter=letter,
            )
        )
    return tuple(scores)


class IndependentMediastinalBodypartVerifier:
    """Ground mediastinum and thoracic-width candidates independently."""

    name = "independent_chestx_det_mediastinal_bodypart_v1"

    def __init__(
        self,
        *,
        prediction: ChestXDetPrediction,
        min_mediastinum_dice: float,
        min_thoracic_line_score: float,
        max_negative_mediastinum_dice: float,
        max_negative_thoracic_line_score: float,
    ) -> None:
        values = (
            min_mediastinum_dice,
            min_thoracic_line_score,
            max_negative_mediastinum_dice,
            max_negative_thoracic_line_score,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError("Mediastinal Stage-2 thresholds must lie in [0, 1]")
        if max_negative_mediastinum_dice >= min_mediastinum_dice:
            raise ValueError("Negative mediastinum Dice must be below its positive Dice")
        if max_negative_thoracic_line_score >= min_thoracic_line_score:
            raise ValueError("Negative thoracic IoU must be below its positive IoU")
        self.prediction = prediction
        self.min_mediastinum_dice = float(min_mediastinum_dice)
        self.min_thoracic_line_score = float(min_thoracic_line_score)
        self.max_negative_mediastinum_dice = float(max_negative_mediastinum_dice)
        self.max_negative_thoracic_line_score = float(
            max_negative_thoracic_line_score
        )

    def _classify(self, score: MediastinalCandidateEvidence) -> str | None:
        mediastinum = score.mediastinum_dice >= self.min_mediastinum_dice
        thoracic = score.thoracic_line_score >= self.min_thoracic_line_score
        if mediastinum and not thoracic:
            return "mediastinum"
        if thoracic and not mediastinum:
            return "thoracic_width"
        if (
            score.mediastinum_dice <= self.max_negative_mediastinum_dice
            and score.thoracic_line_score
            <= self.max_negative_thoracic_line_score
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
                reason="independent_mediastinal_bodypart_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        options = parse_mcq_options(turn.question)
        selected = selected_option_letters(turn.question, response)
        if not selected or any(letter not in options for letter in selected):
            return GateResult(
                passed=False,
                score=0.0,
                reason="mediastinal_bodypart_has_no_valid_selection",
                metadata={
                    "verifier": self.name,
                    "decision": "REPAIR",
                    "selected": list(selected),
                },
            )
        scores = score_mediastinal_candidates(turn, self.prediction)
        classifications = {score.letter: self._classify(score) for score in scores}
        metadata = {
            "verifier": self.name,
            "selected": list(selected),
            "candidate_scores": [
                {
                    **score.__dict__,
                    "classification": classifications[score.letter],
                }
                for score in scores
            ],
            "policy": {
                "min_mediastinum_dice": self.min_mediastinum_dice,
                "min_thoracic_line_score": self.min_thoracic_line_score,
                "max_negative_mediastinum_dice": self.max_negative_mediastinum_dice,
                "max_negative_thoracic_line_score": (
                    self.max_negative_thoracic_line_score
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
            for anatomy in ("mediastinum", "thoracic_width")
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
                else "selection_disagrees_with_independent_mediastinal_anatomy"
            ),
            metadata=metadata,
        )

"""Feature-aligned calibration for the independent bilateral-lung verifier.

CheXmask supplies HybridGNet masks and an RCA quality score.  This module
keeps those masks separate from the ChestX-Det prediction used by the runtime
verifier, creates deterministic positive and hard-negative candidate sets,
and selects a conservative Dice/margin policy on calibration patients only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from cxreason.calibration.inclusion import SplitName, wilson_interval
from cxreason.path1.independent_verifiers import dice_score, iou_score
from cxreason.vision.chestx_det import ChestXDetPrediction


Decision = Literal["candidate", "special", "abstain"]


def decode_rle(rle: str, height: int, width: int) -> np.ndarray:
    """Decode CheXmask's one-indexed row-major run-length encoding."""

    if height <= 0 or width <= 0:
        raise ValueError("Mask dimensions must be positive")
    values = np.fromstring(rle.strip(), sep=" ", dtype=np.int64)
    if values.size == 0 or values.size % 2:
        raise ValueError("RLE must contain non-empty start/length pairs")
    starts = values[0::2] - 1
    lengths = values[1::2]
    if np.any(starts < 0) or np.any(lengths <= 0):
        raise ValueError("RLE starts are one-indexed and lengths must be positive")
    ends = starts + lengths
    if np.any(ends > height * width):
        raise ValueError("RLE run extends beyond the requested mask")
    mask = np.zeros(height * width, dtype=bool)
    for start, end in zip(starts, ends):
        mask[int(start) : int(end)] = True
    return mask.reshape((height, width))


@dataclass(frozen=True)
class CheXmaskLungAnnotation:
    image_file: str
    patient_id: str
    split: SplitName
    dice_rca_mean: float
    height: int
    width: int
    left_lung_rle: str
    right_lung_rle: str
    heart_rle: str

    def __post_init__(self) -> None:
        if not self.image_file or not self.patient_id:
            raise ValueError("Image and patient identifiers must be non-empty")
        if not math.isfinite(self.dice_rca_mean) or not 0.0 <= self.dice_rca_mean <= 1.0:
            raise ValueError("dice_rca_mean must lie in [0, 1]")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Mask dimensions must be positive")

    def masks(self) -> dict[str, np.ndarray]:
        left = decode_rle(self.left_lung_rle, self.height, self.width)
        right = decode_rle(self.right_lung_rle, self.height, self.width)
        heart = decode_rle(self.heart_rle, self.height, self.width)
        return {
            "bilateral_lungs": left | right,
            "left_lung": left,
            "right_lung": right,
            "heart": heart,
            "heart_plus_left_lung": heart | left,
        }


@dataclass(frozen=True)
class CandidateMaskScore:
    name: str
    dice: float
    iou: float


@dataclass(frozen=True)
class LungCalibrationRecord:
    image_file: str
    patient_id: str
    split: SplitName
    dice_rca_mean: float
    positive_scores: tuple[CandidateMaskScore, ...]
    special_scores: tuple[CandidateMaskScore, ...]


def score_chexmask_annotation(
    annotation: CheXmaskLungAnnotation,
    prediction: ChestXDetPrediction,
) -> LungCalibrationRecord:
    """Score exact runtime Dice features against independent CheXmask masks.

    The positive set mirrors a five-option native turn: four image candidates
    plus a special option.  The special set deliberately omits bilateral lungs
    and contains unilateral/hybrid hard negatives, not merely easy empty masks.
    """

    if prediction.original_size != (annotation.width, annotation.height):
        raise ValueError(
            "CheXmask dimensions do not match the source image: "
            f"{(annotation.width, annotation.height)} != {prediction.original_size}"
        )
    independent = prediction.project_mask(
        ("Left Lung", "Right Lung"), (annotation.width, annotation.height)
    )
    masks = annotation.masks()

    def score(names: Iterable[str]) -> tuple[CandidateMaskScore, ...]:
        return tuple(
            CandidateMaskScore(
                name=name,
                dice=dice_score(independent, masks[name]),
                iou=iou_score(independent, masks[name]),
            )
            for name in names
        )

    return LungCalibrationRecord(
        image_file=annotation.image_file,
        patient_id=annotation.patient_id,
        split=annotation.split,
        dice_rca_mean=annotation.dice_rca_mean,
        positive_scores=score(
            ("bilateral_lungs", "heart", "left_lung", "right_lung")
        ),
        special_scores=score(
            ("heart", "left_lung", "right_lung", "heart_plus_left_lung")
        ),
    )


def policy_decision(
    scores: Sequence[CandidateMaskScore],
    *,
    min_dice: float,
    max_special_dice: float,
    min_dice_margin: float,
) -> tuple[Decision, str | None]:
    if not scores:
        return "special", None
    ranked = sorted(scores, key=lambda value: (-value.dice, value.name))
    if ranked[0].dice <= max_special_dice:
        return "special", None
    if ranked[0].dice < min_dice:
        return "abstain", None
    if len(ranked) > 1 and ranked[0].dice - ranked[1].dice < min_dice_margin:
        return "abstain", None
    return "candidate", ranked[0].name


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def evaluate_lung_policy(
    records: Sequence[LungCalibrationRecord],
    *,
    split: SplitName,
    min_dice: float,
    max_special_dice: float,
    min_dice_margin: float,
) -> dict[str, object]:
    if not 0.0 < min_dice < 1.0:
        raise ValueError("min_dice must lie in (0, 1)")
    if not 0.0 <= min_dice_margin < 1.0:
        raise ValueError("min_dice_margin must lie in [0, 1)")
    if not 0.0 <= max_special_dice < min_dice:
        raise ValueError("max_special_dice must lie in [0, min_dice)")
    counts = {
        "positive_correct": 0,
        "positive_wrong": 0,
        "positive_abstain": 0,
        "special_correct": 0,
        "special_wrong": 0,
        "special_abstain": 0,
    }
    selected = [record for record in records if record.split == split]
    for record in selected:
        decision, candidate = policy_decision(
            record.positive_scores,
            min_dice=min_dice,
            max_special_dice=max_special_dice,
            min_dice_margin=min_dice_margin,
        )
        if decision == "abstain":
            counts["positive_abstain"] += 1
        elif decision == "candidate" and candidate == "bilateral_lungs":
            counts["positive_correct"] += 1
        else:
            counts["positive_wrong"] += 1

        decision, _ = policy_decision(
            record.special_scores,
            min_dice=min_dice,
            max_special_dice=max_special_dice,
            min_dice_margin=min_dice_margin,
        )
        if decision == "abstain":
            counts["special_abstain"] += 1
        elif decision == "special":
            counts["special_correct"] += 1
        else:
            counts["special_wrong"] += 1

    correct = counts["positive_correct"] + counts["special_correct"]
    wrong = counts["positive_wrong"] + counts["special_wrong"]
    abstain = counts["positive_abstain"] + counts["special_abstain"]
    decisions = correct + wrong
    total = decisions + abstain
    error_interval = wilson_interval(wrong, decisions)
    return {
        "split": split,
        "patients": len({record.patient_id for record in selected}),
        "images": len(selected),
        "examples": total,
        "min_dice": float(min_dice),
        "max_special_dice": float(max_special_dice),
        "min_dice_margin": float(min_dice_margin),
        **counts,
        "correct_decisions": correct,
        "wrong_decisions": wrong,
        "abstentions": abstain,
        "decision_coverage": _ratio(decisions, total),
        "decision_precision": _ratio(correct, decisions),
        "wrong_decision_rate": _ratio(wrong, decisions),
        "wrong_decision_wilson95": (
            [round(value, 6) for value in error_interval]
            if error_interval is not None
            else None
        ),
        "positive_coverage": _ratio(
            counts["positive_correct"] + counts["positive_wrong"], len(selected)
        ),
        "special_coverage": _ratio(
            counts["special_correct"] + counts["special_wrong"], len(selected)
        ),
    }


def select_lung_policy(
    records: Sequence[LungCalibrationRecord],
    *,
    dice_candidates: Sequence[float],
    special_dice_candidates: Sequence[float],
    margin_candidates: Sequence[float],
    require_zero_wrong_decisions: bool = True,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Select the highest-coverage safe policy using calibration patients only."""

    dice_values = sorted(set(float(value) for value in dice_candidates))
    special_values = sorted(set(float(value) for value in special_dice_candidates))
    margin_values = sorted(set(float(value) for value in margin_candidates))
    if not dice_values or any(not 0.0 < value < 1.0 for value in dice_values):
        raise ValueError("Dice candidates must lie in (0, 1)")
    if not margin_values or any(not 0.0 <= value < 1.0 for value in margin_values):
        raise ValueError("Margin candidates must lie in [0, 1)")
    if not special_values or any(not 0.0 <= value < 1.0 for value in special_values):
        raise ValueError("Special Dice candidates must lie in [0, 1)")
    sweep = [
        evaluate_lung_policy(
            records,
            split="calibration",
            min_dice=dice,
            max_special_dice=special,
            min_dice_margin=margin,
        )
        for dice in dice_values
        for special in special_values
        if special < dice
        for margin in margin_values
    ]
    eligible = [value for value in sweep if value["wrong_decisions"] == 0]
    if not eligible and require_zero_wrong_decisions:
        raise RuntimeError("No zero-wrong-decision calibration policy exists")
    pool = eligible or sweep

    def key(value: dict[str, object]) -> tuple[float, ...]:
        positive = int(value["positive_correct"])
        special = int(value["special_correct"])
        return (
            float(value["correct_decisions"]),
            float(min(positive, special)),
            float(positive),
            -float(value["wrong_decisions"]),
            float(value["min_dice"]) - float(value["max_special_dice"]),
            float(value["min_dice_margin"]),
        )

    choice = max(pool, key=key)
    return {
        "min_dice": float(choice["min_dice"]),
        "max_special_dice": float(choice["max_special_dice"]),
        "min_dice_margin": float(choice["min_dice_margin"]),
    }, sweep

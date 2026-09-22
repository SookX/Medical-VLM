"""Patient-level calibration helpers for independent cardiothoracic ratio."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, Literal

import numpy as np


Split = Literal["calibration", "validation", "test"]


@dataclass(frozen=True)
class CardiothoracicCalibrationRow:
    image_file: str
    patient_id: str
    split: Split
    raw_ctr: float
    reference_ctr: float


def reference_interval(ctr: float) -> tuple[float, float]:
    """Return the three-hundredth option centered on the rounded CTR."""

    rounded = Decimal(str(float(ctr))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    center = float(rounded)
    return (round(center - 0.01, 2), round(center + 0.01, 2))


def fit_affine_ctr(
    rows: Iterable[CardiothoracicCalibrationRow],
) -> tuple[float, float]:
    """Fit reference CTR = intercept + slope * independent mask CTR."""

    selected = [row for row in rows if row.split == "calibration"]
    if len(selected) < 2:
        raise ValueError("At least two calibration rows are required")
    x = np.asarray([row.raw_ctr for row in selected], dtype=float)
    y = np.asarray([row.reference_ctr for row in selected], dtype=float)
    design = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope <= 0:
        raise ValueError("Independent CTR calibration produced an invalid affine fit")
    return float(intercept), float(slope)


def calibration_residual_interval(
    rows: Iterable[CardiothoracicCalibrationRow],
    *,
    intercept: float,
    slope: float,
    safety_padding: float,
) -> tuple[float, float]:
    """Lock an extrema-based residual interval using calibration rows only."""

    if safety_padding < 0:
        raise ValueError("safety_padding must be non-negative")
    residuals = [
        row.reference_ctr - (intercept + slope * row.raw_ctr)
        for row in rows
        if row.split == "calibration"
    ]
    if not residuals:
        raise ValueError("No calibration residuals are available")
    return (
        float(min(residuals) - safety_padding),
        float(max(residuals) + safety_padding),
    )


def plausible_ctr_interval(
    raw_ctr: float,
    *,
    intercept: float,
    slope: float,
    residual_low: float,
    residual_high: float,
) -> tuple[float, float]:
    estimate = intercept + slope * raw_ctr
    return (
        max(0.0, estimate + residual_low),
        min(1.0, estimate + residual_high),
    )


def intervals_overlap(
    first: tuple[float, float], second: tuple[float, float]
) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1]) + 1e-12


def evaluate_ctr_policy(
    rows: Iterable[CardiothoracicCalibrationRow],
    *,
    split: Split,
    intercept: float,
    slope: float,
    residual_low: float,
    residual_high: float,
) -> dict[str, object]:
    selected = [row for row in rows if row.split == split]
    false_repairs = 0
    alternatives = 0
    alternatives_detected = 0
    widths: list[float] = []
    absolute_errors: list[float] = []
    for row in selected:
        plausible = plausible_ctr_interval(
            row.raw_ctr,
            intercept=intercept,
            slope=slope,
            residual_low=residual_low,
            residual_high=residual_high,
        )
        reference = reference_interval(row.reference_ctr)
        false_repairs += not intervals_overlap(plausible, reference)
        widths.append(plausible[1] - plausible[0])
        estimate = intercept + slope * row.raw_ctr
        absolute_errors.append(abs(row.reference_ctr - estimate))
        center = sum(reference) / 2
        for offset in (-0.06, -0.03, 0.03, 0.06):
            alternative = (round(center + offset - 0.01, 2), round(center + offset + 0.01, 2))
            alternatives += 1
            alternatives_detected += not intervals_overlap(plausible, alternative)
    count = len(selected)
    return {
        "split": split,
        "patients": len({row.patient_id for row in selected}),
        "images": count,
        "correct_reference_options_passed": count - false_repairs,
        "false_repairs": false_repairs,
        "false_repair_rate": round(false_repairs / count, 6) if count else None,
        "alternative_options": alternatives,
        "alternative_options_detected": alternatives_detected,
        "alternative_detection_rate": (
            round(alternatives_detected / alternatives, 6) if alternatives else None
        ),
        "mean_plausible_interval_width": (
            round(float(np.mean(widths)), 6) if widths else None
        ),
        "mean_absolute_error": (
            round(float(np.mean(absolute_errors)), 6) if absolute_errors else None
        ),
        "max_absolute_error": max(absolute_errors) if absolute_errors else None,
    }

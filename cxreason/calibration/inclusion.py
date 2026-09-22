"""Patient-disjoint calibration for the one-sided inclusion geometry gate."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence


SplitName = Literal["calibration", "validation", "test"]
INCLUSION_REGIONS = (
    "apex_right",
    "apex_left",
    "side_right",
    "side_left",
    "bottom_right",
    "bottom_left",
)
_NIH_IMAGE_RE = re.compile(r"^(?P<patient>\d{8})_\d+$")


@dataclass(frozen=True)
class PatientSplitConfig:
    salt: str = "cxreason-path1-inclusion-calibration-v1"
    calibration_basis_points: int = 7000
    validation_basis_points: int = 1500

    def __post_init__(self) -> None:
        if not self.salt:
            raise ValueError("Split salt must be non-empty")
        if self.calibration_basis_points <= 0:
            raise ValueError("Calibration split must be non-empty")
        if self.validation_basis_points <= 0:
            raise ValueError("Validation split must be non-empty")
        if self.calibration_basis_points + self.validation_basis_points >= 10000:
            raise ValueError("Test split must be non-empty")

    @property
    def test_basis_points(self) -> int:
        return 10000 - self.calibration_basis_points - self.validation_basis_points


@dataclass(frozen=True)
class InclusionCalibrationRow:
    image_file: str
    patient_id: str
    split: SplitName
    ratios: tuple[float, ...]
    labels: tuple[bool, ...]

    def ratio(self, region: str) -> float:
        return self.ratios[INCLUSION_REGIONS.index(region)]

    def label(self, region: str) -> bool:
        return self.labels[INCLUSION_REGIONS.index(region)]


def nih_patient_id(image_file: str) -> str:
    """Extract the NIH-CXR14 patient identifier without accepting ambiguity."""

    match = _NIH_IMAGE_RE.fullmatch(image_file.strip())
    if match is None:
        raise ValueError(f"Invalid NIH-CXR14 image identifier: {image_file!r}")
    return match.group("patient")


def assign_patient_split(
    patient_id: str, config: PatientSplitConfig = PatientSplitConfig()
) -> SplitName:
    """Assign a patient deterministically; every image for that patient follows."""

    digest = hashlib.sha256(f"{config.salt}:{patient_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10000
    if bucket < config.calibration_basis_points:
        return "calibration"
    if bucket < config.calibration_basis_points + config.validation_basis_points:
        return "validation"
    return "test"


def load_nih_inclusion_rows(
    path: str | Path,
    config: PatientSplitConfig = PatientSplitConfig(),
) -> list[InclusionCalibrationRow]:
    """Load and validate the independent NIH inclusion table."""

    source = Path(path)
    rows: list[InclusionCalibrationRow] = []
    seen_images: set[str] = set()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_file"}
        for region in INCLUSION_REGIONS:
            required.add(f"ratio_{region}_lung")
            required.add(f"label_{region}_lung")
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing inclusion columns: {sorted(missing)}")
        for line_number, value in enumerate(reader, start=2):
            image_file = value["image_file"].strip()
            if image_file in seen_images:
                raise ValueError(f"Duplicate image_file at line {line_number}: {image_file}")
            seen_images.add(image_file)
            patient_id = nih_patient_id(image_file)
            ratios: list[float] = []
            labels: list[bool] = []
            for region in INCLUSION_REGIONS:
                ratio = float(value[f"ratio_{region}_lung"])
                if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                    raise ValueError(
                        f"Invalid ratio for {region} at line {line_number}: {ratio}"
                    )
                label = value[f"label_{region}_lung"].strip()
                if label not in {"0", "1"}:
                    raise ValueError(
                        f"Invalid label for {region} at line {line_number}: {label!r}"
                    )
                ratios.append(ratio)
                labels.append(label == "1")
            rows.append(
                InclusionCalibrationRow(
                    image_file=image_file,
                    patient_id=patient_id,
                    split=assign_patient_split(patient_id, config),
                    ratios=tuple(ratios),
                    labels=tuple(labels),
                )
            )
    if not rows:
        raise ValueError(f"No inclusion rows found in {source}")
    return rows


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


def _rounded_ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def evaluate_region_threshold(
    rows: Iterable[InclusionCalibrationRow],
    *,
    split: SplitName,
    region: str,
    threshold: float,
) -> dict[str, object]:
    if region not in INCLUSION_REGIONS:
        raise ValueError(f"Unknown inclusion region: {region}")
    counts = Counter()
    patients: set[str] = set()
    for row in rows:
        if row.split != split:
            continue
        patients.add(row.patient_id)
        included = row.label(region)
        certified = row.ratio(region) >= threshold
        counts[
            "included_certified"
            if included and certified
            else "excluded_certified"
            if not included and certified
            else "included_not_certified"
            if included
            else "excluded_not_certified"
        ] += 1
    tp = counts["included_certified"]
    fp = counts["excluded_certified"]
    fn = counts["included_not_certified"]
    tn = counts["excluded_not_certified"]
    false_interval = wilson_interval(fp, fp + tn)
    return {
        "region": region,
        "threshold": threshold,
        "patients": len(patients),
        "images": tp + fp + fn + tn,
        "included_certified": tp,
        "excluded_certified": fp,
        "included_not_certified": fn,
        "excluded_not_certified": tn,
        "false_certification_rate": _rounded_ratio(fp, fp + tn),
        "false_certification_wilson95": (
            [round(value, 6) for value in false_interval]
            if false_interval is not None
            else None
        ),
        "included_certification_rate": _rounded_ratio(tp, tp + fn),
        "certified_precision": _rounded_ratio(tp, tp + fp),
        "certification_coverage": _rounded_ratio(tp + fp, tp + fp + fn + tn),
    }


def calibrate_region_thresholds(
    rows: Sequence[InclusionCalibrationRow],
    candidates: Sequence[float],
    *,
    require_zero_false_certifications: bool = True,
) -> tuple[dict[str, float], dict[str, list[dict[str, object]]]]:
    """Select the most sensitive safe threshold independently per region."""

    normalized = sorted(set(float(value) for value in candidates))
    if not normalized or any(not 0.0 < value < 0.5 for value in normalized):
        raise ValueError("Candidate thresholds must be values between 0 and 0.5")
    selected: dict[str, float] = {}
    sweeps: dict[str, list[dict[str, object]]] = {}
    for region in INCLUSION_REGIONS:
        metrics = [
            evaluate_region_threshold(
                rows,
                split="calibration",
                region=region,
                threshold=threshold,
            )
            for threshold in normalized
        ]
        eligible = [
            value
            for value in metrics
            if value["excluded_certified"] == 0
            and value["excluded_not_certified"] > 0
        ]
        if not eligible and require_zero_false_certifications:
            raise RuntimeError(f"No zero-false-certification threshold for {region}")
        pool = eligible or metrics
        choice = max(
            pool,
            key=lambda value: (
                float(value["included_certification_rate"] or 0.0),
                -int(value["excluded_certified"]),
                -float(value["threshold"]),
            ),
        )
        selected[region] = float(choice["threshold"])
        sweeps[region] = metrics
    return selected, sweeps


def evaluate_policy(
    rows: Sequence[InclusionCalibrationRow],
    *,
    split: SplitName,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    if set(thresholds) != set(INCLUSION_REGIONS):
        raise ValueError("Threshold policy must contain all and only inclusion regions")
    per_region = {
        region: evaluate_region_threshold(
            rows,
            split=split,
            region=region,
            threshold=float(thresholds[region]),
        )
        for region in INCLUSION_REGIONS
    }
    totals = Counter()
    for metrics in per_region.values():
        for key in (
            "included_certified",
            "excluded_certified",
            "included_not_certified",
            "excluded_not_certified",
        ):
            totals[key] += int(metrics.get(key, 0))
    tp, fp = totals["included_certified"], totals["excluded_certified"]
    fn, tn = totals["included_not_certified"], totals["excluded_not_certified"]
    false_interval = wilson_interval(fp, fp + tn)
    split_rows = [row for row in rows if row.split == split]
    return {
        "split": split,
        "patients": len({row.patient_id for row in split_rows}),
        "images": len(split_rows),
        "thresholds": {
            region: float(thresholds[region]) for region in INCLUSION_REGIONS
        },
        "per_region": per_region,
        "aggregate": {
            **dict(totals),
            "false_certification_rate": _rounded_ratio(fp, fp + tn),
            "false_certification_wilson95": (
                [round(value, 6) for value in false_interval]
                if false_interval is not None
                else None
            ),
            "included_certification_rate": _rounded_ratio(tp, tp + fn),
            "certified_precision": _rounded_ratio(tp, tp + fp),
            "certification_coverage": _rounded_ratio(
                tp + fp, tp + fp + fn + tn
            ),
        },
    }

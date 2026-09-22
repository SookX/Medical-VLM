"""Reusable externally calibrated selective verifier for Path-1 anatomy panels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cxreason.calibration.selective_knn import SelectiveKnnModel
from cxreason.data.cxreasonbench import NativeStageTurn
from cxreason.gates.base import GateResult
from cxreason.path1.independent_verifiers import extract_red_overlay_mask
from cxreason.path1.practical_verifiers import (
    _normalize,
    parse_mcq_options,
    selected_option_letters,
)
from cxreason.vision.chestx_det import CHESTX_DET_TARGETS, ChestXDetPrediction


STAGE2_REQUIRED_CANDIDATE_TYPES: dict[str, tuple[str, ...]] = {
    "ascending_aorta_enlargement": ("ascending_aorta", "borderline"),
    "rotation": ("clavicle_both", "midline"),
    "carina_angle": ("carina",),
    "aortic_knob_enlargement": ("aortic_knob", "trachea"),
    "descending_aorta_enlargement": ("trachea", "descending_aorta"),
    "descending_aorta_tortuous": ("descending_aorta",),
    "inspiration": (
        "right_posterior_rib",
        "diaphragm_right",
        "midclavicularline",
    ),
    "trachea_deviation": ("trachea", "midline"),
}


def _longest_run_fraction(mask: np.ndarray, axis: int) -> float:
    working = mask if axis == 1 else mask.T
    best = 0
    for row_index in np.flatnonzero(working.any(axis=1)):
        row = np.asarray(working[row_index], dtype=np.int8)
        changes = np.diff(np.pad(row, (1, 1)))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if starts.size:
            best = max(best, int(np.max(ends - starts)))
    return best / working.shape[1]


def feature_names() -> tuple[str, ...]:
    names = [
        "overlay_fraction",
        "bbox_fill_ratio",
        "bbox_width_fraction",
        "bbox_height_fraction",
        "centroid_x_fraction",
        "centroid_y_fraction",
        "longest_horizontal_run_fraction",
        "longest_vertical_run_fraction",
    ]
    for target in CHESTX_DET_TARGETS:
        slug = target.lower().replace(" ", "_")
        names.extend(
            (
                f"{slug}_dice",
                f"{slug}_iou",
                f"{slug}_precision",
                f"{slug}_recall",
                f"{slug}_centroid_dx",
                f"{slug}_centroid_dy",
            )
        )
    return tuple(names)


def candidate_feature_vector(
    overlay: np.ndarray,
    independent_masks: dict[str, np.ndarray],
) -> tuple[float, ...]:
    overlay = np.asarray(overlay, dtype=bool)
    if set(independent_masks) != set(CHESTX_DET_TARGETS):
        raise ValueError("Generic Stage-2 features require all ChestX-Det targets")
    if any(np.asarray(mask).shape != overlay.shape for mask in independent_masks.values()):
        raise ValueError("Candidate and independent masks must share one shape")
    rows, columns = np.where(overlay)
    if not len(columns):
        return tuple(0.0 for _ in feature_names())
    height, width = overlay.shape
    xmin, xmax = int(columns.min()), int(columns.max())
    ymin, ymax = int(rows.min()), int(rows.max())
    bbox_area = (xmax - xmin + 1) * (ymax - ymin + 1)
    overlay_area = int(overlay.sum())
    cx = float(columns.mean() / width)
    cy = float(rows.mean() / height)
    values = [
        overlay_area / (height * width),
        overlay_area / bbox_area,
        (xmax - xmin + 1) / width,
        (ymax - ymin + 1) / height,
        cx,
        cy,
        _longest_run_fraction(overlay, 1),
        _longest_run_fraction(overlay, 0),
    ]
    for target in CHESTX_DET_TARGETS:
        mask = np.asarray(independent_masks[target], dtype=bool)
        target_area = int(mask.sum())
        intersection = int((overlay & mask).sum())
        union = int((overlay | mask).sum())
        mask_rows, mask_columns = np.where(mask)
        target_cx = float(mask_columns.mean() / width) if len(mask_columns) else 0.5
        target_cy = float(mask_rows.mean() / height) if len(mask_rows) else 0.5
        values.extend(
            (
                2.0 * intersection / (overlay_area + target_area)
                if overlay_area + target_area
                else 1.0,
                intersection / union if union else 1.0,
                intersection / overlay_area if overlay_area else 0.0,
                intersection / target_area if target_area else 0.0,
                abs(cx - target_cx),
                abs(cy - target_cy),
            )
        )
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class ClassifiedCandidate:
    letter: str
    classification: str | None
    distance: float


class IndependentSelectiveAnatomyVerifier:
    name = "independent_selective_anatomy_knn_v1"

    def __init__(
        self,
        *,
        task: str,
        prediction: ChestXDetPrediction,
        policy: dict[str, object],
        model_path: Path,
    ) -> None:
        expected = STAGE2_REQUIRED_CANDIDATE_TYPES.get(task)
        if expected is None:
            raise ValueError(f"No generic Stage-2 task definition: {task}")
        if tuple(policy["required_types"]) != expected:
            raise ValueError("Policy required-type inventory mismatch")
        self.task = task
        self.required_types = expected
        self.prediction = prediction
        self.model = SelectiveKnnModel(model_path, policy)

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

    def _classify_turn(self, turn: NativeStageTurn) -> list[ClassifiedCandidate]:
        output = []
        for index, image_path in enumerate(turn.image_paths):
            letter = chr(ord("a") + index)
            overlay = extract_red_overlay_mask(image_path)
            if overlay is None:
                output.append(ClassifiedCandidate(letter, None, float("inf")))
                continue
            height, width = overlay.shape
            masks = {
                target: self.prediction.project_mask(target, (width, height))
                for target in CHESTX_DET_TARGETS
            }
            label, distance = self.model.predict(
                candidate_feature_vector(overlay, masks)
            )
            output.append(ClassifiedCandidate(letter, label, distance))
        return output

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
                reason="selective_anatomy_verifier_used_for_wrong_stage",
                metadata={"verifier": self.name, "decision": "REPAIR"},
            )
        options = parse_mcq_options(turn.question)
        selected = selected_option_letters(turn.question, response)
        if not selected or any(letter not in options for letter in selected):
            return GateResult(
                passed=False,
                score=0.0,
                reason="bodypart_has_no_valid_selection",
                metadata={
                    "verifier": self.name,
                    "task": self.task,
                    "decision": "REPAIR",
                    "selected": list(selected),
                },
            )
        candidates = self._classify_turn(turn)
        metadata = {
            "verifier": self.name,
            "task": self.task,
            "selected": list(selected),
            "candidate_scores": [candidate.__dict__ for candidate in candidates],
        }
        if any(candidate.classification is None for candidate in candidates):
            metadata.update(
                {"decision": "ABSTAIN", "abstain_reason": "uncertain_candidate"}
            )
            return GateResult(passed=True, metadata=metadata)
        positive = {
            anatomy: sorted(
                candidate.letter
                for candidate in candidates
                if candidate.classification == anatomy
            )
            for anatomy in self.required_types
        }
        if any(len(letters) > 1 for letters in positive.values()):
            metadata.update(
                {"decision": "ABSTAIN", "abstain_reason": "duplicate_anatomy_match"}
            )
            return GateResult(passed=True, metadata=metadata)
        relevant = sorted(letter for letters in positive.values() for letter in letters)
        special = self._special_letters(turn.question)
        if "need_new" in special:
            expected = relevant + (
                [special["need_new"]]
                if any(not positive[name] for name in self.required_types)
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
            reason=None if passed else "selection_disagrees_with_selective_anatomy",
            metadata=metadata,
        )

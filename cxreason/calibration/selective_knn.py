"""Patient-disjoint selective k-nearest-neighbour calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SelectiveFeatureRow:
    image_file: str
    patient_id: str
    split: str
    candidate_type: str
    features: tuple[float, ...]


def expected_label(candidate_type: str, required_types: tuple[str, ...]) -> str:
    return candidate_type if candidate_type in required_types else "negative"


def _nearest_vote(
    query: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[str | None, float]:
    distances = np.sqrt(np.mean(np.square(reference - query), axis=1))
    if distances.size < k:
        return None, float("inf")
    nearest = np.argpartition(distances, k - 1)[:k]
    nearest_labels = labels[nearest]
    if len(set(nearest_labels.tolist())) != 1:
        return None, float(np.max(distances[nearest]))
    return str(nearest_labels[0]), float(np.max(distances[nearest]))


def _select_thresholds(
    actual: list[str],
    predicted: list[str | None],
    distances: list[float],
    classes: tuple[str, ...],
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for label in classes:
        wrong = [
            distance
            for truth, guess, distance in zip(actual, predicted, distances)
            if guess == label and truth != label
        ]
        wrong_limit = min(wrong, default=float("inf"))
        correct = sorted(
            distance
            for truth, guess, distance in zip(actual, predicted, distances)
            if guess == label and truth == label and distance < wrong_limit
        )
        if correct:
            thresholds[label] = float(correct[-1])
    return thresholds


def _metrics(actual: list[str], predicted: list[str | None]) -> dict[str, object]:
    classes = tuple(dict.fromkeys(actual))
    correct = sum(a == p for a, p in zip(actual, predicted))
    wrong = sum(p is not None and a != p for a, p in zip(actual, predicted))
    abstained = sum(p is None for p in predicted)
    per_class = {}
    for label in classes:
        indices = [index for index, value in enumerate(actual) if value == label]
        per_class[label] = {
            "rows": len(indices),
            "correct": sum(predicted[index] == label for index in indices),
            "wrong_conclusive": sum(
                predicted[index] is not None and predicted[index] != label
                for index in indices
            ),
            "abstained": sum(predicted[index] is None for index in indices),
        }
    return {
        "candidate_rows": len(actual),
        "correct_conclusive": correct,
        "wrong_conclusive": wrong,
        "abstained": abstained,
        "conclusive_rate": round((correct + wrong) / len(actual), 6)
        if actual
        else 0.0,
        "per_class": per_class,
    }


def fit_selective_knn(
    rows: list[SelectiveFeatureRow],
    *,
    required_types: tuple[str, ...],
    feature_names: tuple[str, ...],
    model_path: Path,
    k_values: tuple[int, ...] = (1, 3, 5, 7),
) -> tuple[dict[str, object], dict[str, object]]:
    calibration = [row for row in rows if row.split == "calibration"]
    if not calibration:
        raise ValueError("No calibration rows")
    matrix = np.asarray([row.features for row in calibration], dtype=np.float64)
    if matrix.shape[1] != len(feature_names):
        raise ValueError("Feature-name count mismatch")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (matrix - mean) / scale
    labels = np.asarray(
        [expected_label(row.candidate_type, required_types) for row in calibration],
        dtype=str,
    )
    patients = np.asarray([row.patient_id for row in calibration], dtype=str)
    classes = (*required_types, "negative")
    best: tuple[int, int, dict[str, float], list[str | None]] | None = None
    for k in k_values:
        raw_predictions: list[str | None] = []
        raw_distances: list[float] = []
        for index, query in enumerate(standardized):
            keep = patients != patients[index]
            label, distance = _nearest_vote(
                query, standardized[keep], labels[keep], k
            )
            raw_predictions.append(label)
            raw_distances.append(distance)
        thresholds = _select_thresholds(
            labels.tolist(), raw_predictions, raw_distances, classes
        )
        predictions = [
            label
            if label is not None
            and label in thresholds
            and distance <= thresholds[label]
            else None
            for label, distance in zip(raw_predictions, raw_distances)
        ]
        metrics = _metrics(labels.tolist(), predictions)
        if metrics["wrong_conclusive"] != 0:
            raise RuntimeError("Selective calibration produced a wrong mapping")
        candidate = (int(metrics["correct_conclusive"]), k, thresholds, predictions)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None or best[0] == 0:
        raise RuntimeError("No conclusive zero-error selective kNN policy")
    correct, k, thresholds, predictions = best
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        reference=standardized.astype(np.float32),
        labels=labels,
        patient_ids=patients,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        feature_names=np.asarray(feature_names, dtype=str),
    )
    calibration_metrics = _metrics(labels.tolist(), predictions)
    calibration_metrics["patients"] = len(set(patients.tolist()))
    calibration_metrics["evaluation_mode"] = "leave_one_patient_out"
    policy = {
        "classifier": "selective_unanimous_knn_v1",
        "k": k,
        "distance_thresholds": thresholds,
        "required_types": list(required_types),
        "feature_count": len(feature_names),
        "calibration_metrics": calibration_metrics,
    }
    return policy, calibration_metrics


class SelectiveKnnModel:
    def __init__(self, model_path: Path, policy: dict[str, object]) -> None:
        with np.load(model_path, allow_pickle=False) as payload:
            self.reference = payload["reference"].astype(np.float64)
            self.labels = payload["labels"].astype(str)
            self.mean = payload["mean"].astype(np.float64)
            self.scale = payload["scale"].astype(np.float64)
            self.feature_names = tuple(payload["feature_names"].astype(str).tolist())
        raw_indices = policy.get("feature_indices")
        self.feature_indices = (
            tuple(int(value) for value in raw_indices)
            if raw_indices is not None
            else None
        )
        if self.feature_indices is not None and len(self.feature_indices) != len(self.mean):
            raise ValueError("Feature-index count does not match calibrated feature count")
        self.k = int(policy["k"])
        self.thresholds = {
            str(key): float(value)
            for key, value in dict(policy["distance_thresholds"]).items()
        }

    def predict(self, features: tuple[float, ...]) -> tuple[str | None, float]:
        values = np.asarray(features, dtype=np.float64)
        if self.feature_indices is not None:
            values = values[list(self.feature_indices)]
        if values.shape != self.mean.shape:
            raise ValueError("Prediction feature count does not match calibrated model")
        query = (values - self.mean) / self.scale
        label, distance = _nearest_vote(query, self.reference, self.labels, self.k)
        if label is None or distance > self.thresholds.get(label, -1.0):
            return None, distance
        return label, distance


def evaluate_selective_knn(
    rows: list[SelectiveFeatureRow],
    *,
    split: str,
    required_types: tuple[str, ...],
    model: SelectiveKnnModel,
) -> dict[str, object]:
    selected = [row for row in rows if row.split == split]
    actual = [expected_label(row.candidate_type, required_types) for row in selected]
    predicted = [model.predict(row.features)[0] for row in selected]
    metrics = _metrics(actual, predicted)
    metrics["split"] = split
    metrics["patients"] = len({row.patient_id for row in selected})
    return metrics

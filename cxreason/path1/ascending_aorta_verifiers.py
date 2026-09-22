"""Representation-faithful geometry for Ascending-aorta Stage 2."""

from __future__ import annotations

import numpy as np


LABELS = ("negative", "ascending_aorta", "borderline")


def selective_label(probabilities: np.ndarray, *, confidence: float, margin: float) -> str | None:
    values = np.asarray(probabilities, dtype=float)
    order = np.argsort(values)[::-1]
    if values[order[0]] < confidence or values[order[0]] - values[order[1]] < margin:
        return None
    return LABELS[int(order[0])]


def draw_reference_line(shape: tuple[int, int], heart: tuple[float, float], trachea: tuple[float, float]) -> np.ndarray:
    """Rasterize normalized heart-to-trachea endpoints with thin end caps."""

    height, width = shape
    x0, y0 = heart[0] * (width - 1), heart[1] * (height - 1)
    x1, y1 = trachea[0] * (width - 1), trachea[1] * (height - 1)
    steps = max(2, int(round(max(abs(x1 - x0), abs(y1 - y0)))) + 1)
    xs = np.rint(np.linspace(x0, x1, steps)).astype(int)
    ys = np.rint(np.linspace(y0, y1, steps)).astype(int)
    result = np.zeros(shape, dtype=bool)
    thickness = max(1, round(min(shape) / 512))
    for x, y in zip(xs, ys):
        result[max(0, y - thickness):min(height, y + thickness + 1), max(0, x - thickness):min(width, x + thickness + 1)] = True
    return result


def reference_line_features(overlay: np.ndarray, prediction: dict[str, float]) -> np.ndarray:
    mask = np.asarray(overlay, dtype=bool)
    predicted_line = draw_reference_line(mask.shape, (prediction["heart_x"], prediction["heart_y"]), (prediction["trachea_x"], prediction["trachea_y"]))
    intersection = int((mask & predicted_line).sum()); union = int((mask | predicted_line).sum())
    rows, columns = np.where(mask)
    if not len(rows):
        return np.zeros(10, dtype=np.float64)
    height, width = mask.shape
    return np.asarray([
        intersection / union if union else 0.0,
        intersection / max(1, int(mask.sum())),
        intersection / max(1, int(predicted_line.sum())),
        float(mask.mean()),
        (columns.max() - columns.min() + 1) / width,
        (rows.max() - rows.min() + 1) / height,
        columns.mean() / width,
        rows.mean() / height,
        prediction["heart_y"] - prediction["trachea_y"],
        abs(prediction["heart_x"] - prediction["trachea_x"]),
    ], dtype=np.float64)

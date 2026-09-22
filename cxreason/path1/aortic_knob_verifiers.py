"""Landmark-grounded candidate features for Aortic-knob Stage 2."""

from __future__ import annotations

import numpy as np


LABELS = ("negative", "aortic_knob", "trachea")


def landmark_candidate_features(overlay: np.ndarray, prediction: dict[str, float]) -> np.ndarray:
    """Describe a displayed candidate relative to predicted Task-1 landmarks."""

    mask = np.asarray(overlay, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return np.zeros(22, dtype=np.float64)
    height, width = mask.shape
    rows, columns = np.where(mask)
    xmin, xmax = columns.min() / width, columns.max() / width
    ymin, ymax = rows.min() / height, rows.max() / height
    bbox_width = xmax - xmin + 1 / width
    bbox_height = ymax - ymin + 1 / height
    target_y = 0.5 * (prediction["trachea_left_y"] + prediction["trachea_right_y"])
    occupied_rows = np.flatnonzero(mask.any(axis=1))
    nearest_row = int(occupied_rows[np.argmin(np.abs(occupied_rows / height - target_y))])
    row_columns = np.flatnonzero(mask[nearest_row])
    row_left, row_right = row_columns[0] / width, row_columns[-1] / width
    knob_error = abs(xmin - prediction["knob_left_x"]) + abs(xmax - prediction["knob_right_x"])
    trachea_row_error = (
        abs(row_left - prediction["trachea_left_x"])
        + abs(row_right - prediction["trachea_right_x"])
        + abs(nearest_row / height - target_y)
    )
    trachea_bbox_error = abs(xmin - prediction["trachea_left_x"]) + abs(xmax - prediction["trachea_right_x"])
    cropped = mask[rows.min() : rows.max() + 1, columns.min() : columns.max() + 1]
    return np.asarray(
        [
            xmin, xmax, ymin, ymax, bbox_width, bbox_height,
            bbox_width / max(bbox_height, 1e-6), float(mask.mean()), float(cropped.mean()),
            float(columns.mean() / width), float(rows.mean() / height),
            row_left, row_right, float(nearest_row / height),
            knob_error, trachea_row_error, trachea_bbox_error,
            prediction["knob_left_x"], prediction["knob_right_x"],
            prediction["trachea_left_x"], prediction["trachea_right_x"], target_y,
        ],
        dtype=np.float64,
    )


def selective_label(probabilities: np.ndarray, *, confidence: float, margin: float) -> str | None:
    values = np.asarray(probabilities, dtype=float)
    order = np.argsort(values)[::-1]
    if values[order[0]] < confidence or values[order[0]] - values[order[1]] < margin:
        return None
    return LABELS[int(order[0])]


def scale_robust_shape_label(overlay: np.ndarray) -> str:
    """Classify only clinically distinctive overlay geometry, else negative.

    The rules avoid absolute landmark predictions and tolerate display rescaling.
    They are deliberately conservative: a short dense upper-mediastinal blob is
    an aortic-knob candidate; a tall narrow upper-midline mask is a trachea.
    """
    mask = np.asarray(overlay, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return "negative"
    height, width = mask.shape; rows, columns = np.where(mask)
    bbox_width = (columns.max() - columns.min() + 1) / width
    bbox_height = (rows.max() - rows.min() + 1) / height
    aspect = bbox_width / max(bbox_height, 1e-6)
    crop = mask[rows.min():rows.max()+1, columns.min():columns.max()+1]
    density = float(crop.mean()); center_x = float(columns.mean()/width); center_y = float(rows.mean()/height)
    if (0.70 <= aspect <= 1.80 and density >= 0.50 and 0.42 <= center_x <= 0.68
            and 0.12 <= center_y <= 0.35 and bbox_width <= 0.22 and bbox_height <= 0.20):
        return "aortic_knob"
    if (0.28 <= aspect <= 0.80 and 0.15 <= density <= 0.90 and 0.38 <= center_x <= 0.62
            and 0.08 <= center_y <= 0.34 and 0.12 <= bbox_height <= 0.55 and bbox_width <= 0.25):
        return "trachea"
    return "negative"


def descending_aorta_shape_label(overlay: np.ndarray) -> str:
    """Conservatively identify displayed trachea or descending-aorta masks."""
    mask=np.asarray(overlay,dtype=bool)
    if mask.ndim!=2 or not mask.any():return "negative"
    height,width=mask.shape;rows,columns=np.where(mask);bbox_width=(columns.max()-columns.min()+1)/width;bbox_height=(rows.max()-rows.min()+1)/height;aspect=bbox_width/max(bbox_height,1e-6);crop=mask[rows.min():rows.max()+1,columns.min():columns.max()+1];density=float(crop.mean());center_x=float(columns.mean()/width);center_y=float(rows.mean()/height)
    if (0.28<=aspect<=0.80 and 0.30<=density<=0.90 and 0.38<=center_x<=0.62 and 0.08<=center_y<=0.34 and 0.10<=bbox_height<=0.55 and bbox_width<=0.25):return "trachea"
    if (0.08<=aspect<=0.50 and 0.25<=density<=0.90 and 0.42<=center_x<=0.72 and 0.48<=center_y<=0.85 and 0.25<=bbox_height<=0.95 and bbox_width<=0.25):return "descending_aorta"
    return "negative"


def mediastinal_shape_label(overlay: np.ndarray) -> str:
    mask=np.asarray(overlay,dtype=bool)
    if mask.ndim!=2 or not mask.any():return "negative"
    h,w=mask.shape;r,c=np.where(mask);bw=(c.max()-c.min()+1)/w;bh=(r.max()-r.min()+1)/h;aspect=bw/max(bh,1e-6);crop=mask[r.min():r.max()+1,c.min():c.max()+1];density=float(crop.mean());cx=float(c.mean()/w);cy=float(r.mean()/h)
    if aspect>=5.0 and bh<=0.14 and 0.35<=bw<=0.72 and 0.32<=cy<=0.68:return "thoracic_width"
    if 0.35<=aspect<=0.90 and 0.25<=density<=0.85 and 0.35<=cx<=0.70 and 0.30<=cy<=0.72 and 0.40<=bh<=0.95:return "mediastinum"
    return "negative"


def rotation_shape_label(overlay: np.ndarray) -> str:
    mask=np.asarray(overlay,dtype=bool)
    if mask.ndim!=2 or not mask.any():return "negative"
    h,w=mask.shape;r,c=np.where(mask);bw=(c.max()-c.min()+1)/w;bh=(r.max()-r.min()+1)/h;aspect=bw/max(bh,1e-6);crop=mask[r.min():r.max()+1,c.min():c.max()+1];density=float(crop.mean());cx=float(c.mean()/w);cy=float(r.mean()/h)
    if aspect>=2.0 and bw>=0.45 and bh<=0.30 and cy<=0.32:return "clavicle_both"
    if aspect<=0.12 and bw<=0.06 and 0.15<=bh<=0.40 and 0.38<=cx<=0.62 and 0.15<=cy<=0.35:return "midline"
    return "negative"

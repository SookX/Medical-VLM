"""Deterministic measurements from CXReasonBench red anatomy overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def horizontal_widths(mask: np.ndarray) -> np.ndarray:
    """Return inclusive horizontal extents for every non-empty mask row."""
    binary = np.asarray(mask, dtype=bool)
    values: list[float] = []
    for row in binary:
        columns = np.flatnonzero(row)
        if len(columns):
            values.append(float(columns[-1] - columns[0] + 1))
    return np.asarray(values, dtype=float)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 8-connected component of a binary overlay."""
    from scipy.ndimage import label

    binary = np.asarray(mask, dtype=bool)
    components, count = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return binary
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def red_overlay_mask_ensemble(image_path: Path) -> list[np.ndarray]:
    """Extract the overlay under fixed increasingly strict color thresholds.

    Returning independently thresholded masks lets callers abstain when JPEG
    blending or underlying anatomy makes the downstream measurement unstable.
    """
    from PIL import Image

    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.int16)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    thresholds = (
        (25, 20, 80),
        (35, 30, 90),
        (45, 40, 105),
        (60, 50, 120),
    )
    masks = []
    for rg, rb, minimum in thresholds:
        mask = largest_component((red - green > rg) & (red - blue > rb) & (red > minimum))
        if mask.any():
            masks.append(mask)
    return masks


def ratio_measurement_ensemble(target_path: Path, trachea_path: Path) -> list[float]:
    """Measure the width ratio across matched overlay thresholds."""
    targets = red_overlay_mask_ensemble(target_path)
    tracheas = red_overlay_mask_ensemble(trachea_path)
    return [maximum_to_median_width_ratio(a, b) for a, b in zip(targets, tracheas)]


def carina_measurement_ensemble(image_path: Path) -> list[float]:
    """Measure Carina angle across the fixed overlay-threshold ensemble."""
    values = []
    for mask in red_overlay_mask_ensemble(image_path):
        for method in (carina_angle_from_mask, carina_angle_extrema, carina_angle_pca):
            try:
                values.append(method(mask))
            except ValueError:
                continue
    return values


def maximum_to_median_width_ratio(target_mask: np.ndarray, trachea_mask: np.ndarray) -> float:
    """CheXStruct ratio: maximum target width / median tracheal width."""
    target = horizontal_widths(target_mask)
    trachea = horizontal_widths(trachea_mask)
    if not len(target) or not len(trachea) or np.median(trachea) <= 0:
        raise ValueError("Both masks must have non-zero horizontal widths")
    return float(target.max() / np.median(trachea))


def carina_angle_from_mask(mask: np.ndarray) -> float:
    """Estimate the included angle of a thick V-shaped carina mask.

    The superior skeleton tip is the bifurcation vertex.  The two inferior
    extremes define the left and right rays; small end bands make the estimate
    insensitive to one-pixel skeletonization spurs.
    """
    from scipy.ndimage import convolve
    from skimage.morphology import skeletonize

    skeleton = skeletonize(np.asarray(mask, dtype=bool))
    if skeleton.sum() < 3:
        raise ValueError("Carina mask has no usable skeleton")
    points = np.argwhere(skeleton).astype(float)
    ymin, ymax = points[:, 0].min(), points[:, 0].max()
    height = max(1.0, ymax - ymin)
    superior = points[points[:, 0] <= ymin + 0.05 * height]
    vertex = np.asarray([superior[:, 0].mean(), np.median(superior[:, 1])])
    neighbours = convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    endpoints = np.argwhere(skeleton & (neighbours == 1)).astype(float)
    inferior = endpoints[endpoints[:, 0] >= ymin + 0.25 * height]
    if len(inferior) < 2:
        raise ValueError("Carina skeleton has no inferior rays")
    left = inferior[np.argmin(inferior[:, 1])]
    right = inferior[np.argmax(inferior[:, 1])]
    a = left - vertex
    b = right - vertex
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def carina_angle_extrema(mask: np.ndarray) -> float:
    """Three-point angle from superior, leftmost, and rightmost mask extrema."""
    points = np.argwhere(np.asarray(mask, dtype=bool)).astype(float)
    if len(points) < 3:
        raise ValueError("Carina mask has fewer than three pixels")
    height = max(1.0, float(np.ptp(points[:, 0])))
    width = max(1.0, float(np.ptp(points[:, 1])))
    top = points[points[:, 0] <= points[:, 0].min() + 0.03 * height]
    left_band = points[points[:, 1] <= points[:, 1].min() + 0.03 * width]
    right_band = points[points[:, 1] >= points[:, 1].max() - 0.03 * width]
    vertex = np.asarray([top[:, 0].mean(), np.median(top[:, 1])])
    left = np.asarray([np.median(left_band[:, 0]), left_band[:, 1].mean()])
    right = np.asarray([np.median(right_band[:, 0]), right_band[:, 1].mean()])
    a, b = left - vertex, right - vertex
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def carina_angle_pca(mask: np.ndarray) -> float:
    """Angle between principal axes of the two inferior Carina branches."""
    points = np.argwhere(np.asarray(mask, dtype=bool)).astype(float)
    if len(points) < 6:
        raise ValueError("Carina mask is too small for PCA")
    ymin, ymax = points[:, 0].min(), points[:, 0].max()
    height = max(1.0, ymax - ymin)
    top = points[points[:, 0] <= ymin + 0.05 * height]
    vertex_x = float(np.median(top[:, 1]))
    lower = points[points[:, 0] >= ymin + 0.15 * height]
    branches = (lower[lower[:, 1] < vertex_x], lower[lower[:, 1] > vertex_x])
    vectors = []
    for branch in branches:
        if len(branch) < 3:
            raise ValueError("Carina mask lacks a separable branch")
        centered = branch - branch.mean(axis=0)
        _values, vectors_matrix = np.linalg.eigh(np.cov(centered.T))
        vector = vectors_matrix[:, -1]
        if vector[0] < 0:
            vector = -vector
        vectors.append(vector)
    cosine = float(np.dot(vectors[0], vectors[1]))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

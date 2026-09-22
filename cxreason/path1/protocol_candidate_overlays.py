"""Protocol-faithful synthetic overlays for externally calibrated anatomy selection."""

from __future__ import annotations

import numpy as np

from scripts.calibrate_path1_cardiomegaly_stage2 import candidate_overlay


def _midclavicular_reference_line(linked) -> np.ndarray:
    """Construct the clinical vertical reference line, not a clavicle mask."""
    clavicle = np.asarray(linked.combined_mask(("clavicle right",)), dtype=bool)
    diaphragm = np.asarray(
        linked.combined_mask(("right hemidiaphragm",)), dtype=bool
    )
    result = np.zeros(clavicle.shape, dtype=bool)
    clavicle_rows, clavicle_columns = np.where(clavicle)
    if not len(clavicle_columns):
        return result
    column = int(round(float(np.median(clavicle_columns))))
    top = int(round(float(np.median(clavicle_rows))))
    diaphragm_rows = np.where(diaphragm)[0]
    bottom = int(diaphragm_rows.max()) if len(diaphragm_rows) else clavicle.shape[0] - 1
    if bottom < top:
        bottom = clavicle.shape[0] - 1
    half_width = max(1, round(clavicle.shape[1] / 256))
    result[top : bottom + 1, max(0, column - half_width) : column + half_width + 1] = True
    return result


def protocol_candidate_overlay(candidate_type: str, linked, annotation) -> np.ndarray:
    """Return v2 overlays while retaining every unaffected v1 representation."""
    if candidate_type == "right_posterior_rib":
        # The inspiration criterion references the tenth visible posterior rib,
        # not the union of all twelve posterior ribs used by the v1 surrogate.
        return linked.combined_mask(("posterior 10th rib right",))
    if candidate_type == "midclavicularline":
        return _midclavicular_reference_line(linked)
    return candidate_overlay(candidate_type, linked, annotation)

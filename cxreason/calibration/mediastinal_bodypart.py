"""External calibration primitives for Mediastinal Widening Stage-2 grounding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MediastinalBodypartRow:
    image_file: str
    patient_id: str
    split: str
    candidate_type: str
    expected_class: str
    mediastinum_dice: float
    thoracic_line_score: float


def classify_mediastinal_candidate(
    row: MediastinalBodypartRow,
    policy: dict[str, float],
) -> str | None:
    mediastinum = row.mediastinum_dice >= float(policy["min_mediastinum_dice"])
    thoracic = row.thoracic_line_score >= float(policy["min_thoracic_line_score"])
    if mediastinum and not thoracic:
        return "mediastinum"
    if thoracic and not mediastinum:
        return "thoracic_width"
    if (
        row.mediastinum_dice <= float(policy["max_negative_mediastinum_dice"])
        and row.thoracic_line_score
        <= float(policy["max_negative_thoracic_line_score"])
    ):
        return "negative"
    return None


def _positive_threshold(
    rows: list[MediastinalBodypartRow],
    *,
    expected: str,
    field: str,
    grid: list[float],
) -> float:
    best: tuple[int, float] | None = None
    for threshold in grid[1:]:
        predicted = [row for row in rows if getattr(row, field) >= threshold]
        if any(row.expected_class != expected for row in predicted):
            continue
        correct = sum(row.expected_class == expected for row in predicted)
        candidate = (correct, -threshold)
        if best is None or candidate > best:
            best = candidate
    if best is None or best[0] == 0:
        raise RuntimeError(f"No zero-error Mediastinal Stage-2 policy for {expected}")
    return -best[1]


def select_mediastinal_policy(
    rows: list[MediastinalBodypartRow], *, grid_step: float = 0.01
) -> dict[str, float]:
    """Maximize conclusive calibration mappings subject to zero wrong mappings."""

    calibration = [row for row in rows if row.split == "calibration"]
    if not calibration:
        raise ValueError("No Mediastinal Stage-2 calibration rows")
    grid = [round(index * grid_step, 6) for index in range(int(1 / grid_step) + 1)]
    min_mediastinum = _positive_threshold(
        calibration,
        expected="mediastinum",
        field="mediastinum_dice",
        grid=grid,
    )
    min_thoracic = _positive_threshold(
        calibration,
        expected="thoracic_width",
        field="thoracic_line_score",
        grid=grid,
    )

    positives = [row for row in calibration if row.expected_class != "negative"]
    negatives = [row for row in calibration if row.expected_class == "negative"]
    best_negative: tuple[int, float, float] | None = None
    for mediastinum_limit in grid:
        if mediastinum_limit >= min_mediastinum:
            continue
        for thoracic_limit in grid:
            if thoracic_limit >= min_thoracic:
                continue
            if any(
                row.mediastinum_dice <= mediastinum_limit
                and row.thoracic_line_score <= thoracic_limit
                for row in positives
            ):
                continue
            correct = sum(
                row.mediastinum_dice <= mediastinum_limit
                and row.thoracic_line_score <= thoracic_limit
                for row in negatives
            )
            candidate = (correct, -mediastinum_limit, -thoracic_limit)
            if best_negative is None or candidate > best_negative:
                best_negative = candidate
    if best_negative is None or best_negative[0] == 0:
        raise RuntimeError("No zero-error Mediastinal Stage-2 negative policy")
    return {
        "min_mediastinum_dice": min_mediastinum,
        "min_thoracic_line_score": min_thoracic,
        "max_negative_mediastinum_dice": -best_negative[1],
        "max_negative_thoracic_line_score": -best_negative[2],
    }


def evaluate_mediastinal_policy(
    rows: list[MediastinalBodypartRow],
    split: str,
    policy: dict[str, float],
) -> dict[str, object]:
    selected = [row for row in rows if row.split == split]
    predictions = [classify_mediastinal_candidate(row, policy) for row in selected]
    expected_classes = ("mediastinum", "thoracic_width", "negative")
    per_class: dict[str, dict[str, int]] = {}
    for expected in expected_classes:
        indices = [
            index for index, row in enumerate(selected) if row.expected_class == expected
        ]
        per_class[expected] = {
            "rows": len(indices),
            "correct": sum(predictions[index] == expected for index in indices),
            "wrong_conclusive": sum(
                predictions[index] is not None and predictions[index] != expected
                for index in indices
            ),
            "abstained": sum(predictions[index] is None for index in indices),
        }
    correct = sum(
        prediction == row.expected_class
        for row, prediction in zip(selected, predictions)
    )
    wrong = sum(
        prediction is not None and prediction != row.expected_class
        for row, prediction in zip(selected, predictions)
    )
    abstained = sum(prediction is None for prediction in predictions)
    return {
        "split": split,
        "patients": len({row.patient_id for row in selected}),
        "candidate_rows": len(selected),
        "correct_conclusive": correct,
        "wrong_conclusive": wrong,
        "abstained": abstained,
        "conclusive_rate": round((correct + wrong) / len(selected), 6)
        if selected
        else 0.0,
        "per_class": per_class,
    }

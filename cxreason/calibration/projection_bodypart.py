"""Leakage-safe calibration for the projection Stage-2 candidate matcher."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionBodypartRow:
    image_file: str
    patient_id: str
    split: str
    candidate_type: str
    expected_class: str
    lung_dice: float
    scapula_dice: float


def classify_projection_candidate(
    row: ProjectionBodypartRow,
    policy: dict[str, dict[str, float]],
) -> str | None:
    values = {"lung": row.lung_dice, "scapula": row.scapula_dice}
    eligible = [
        anatomy
        for anatomy in ("lung", "scapula")
        if values[anatomy] >= policy["min_dice"][anatomy]
        and values[anatomy] - values["scapula" if anatomy == "lung" else "lung"]
        >= policy["min_class_margin"][anatomy]
    ]
    if len(eligible) == 1:
        return eligible[0]
    if all(
        values[anatomy] <= policy["max_negative_dice"][anatomy]
        for anatomy in ("lung", "scapula")
    ):
        return "negative"
    return None


def select_projection_policy(
    rows: list[ProjectionBodypartRow], *, grid_step: float = 0.005
) -> dict[str, dict[str, float]]:
    """Maximize conclusive calibration mappings subject to zero wrong mappings."""

    calibration = [row for row in rows if row.split == "calibration"]
    if not calibration:
        raise ValueError("No projection body-part calibration rows")
    grid = [round(index * grid_step, 6) for index in range(int(1 / grid_step) + 1)]
    min_dice: dict[str, float] = {}
    margins: dict[str, float] = {}
    for anatomy in ("lung", "scapula"):
        other = "scapula" if anatomy == "lung" else "lung"
        best: tuple[int, float, float] | None = None
        for threshold in grid[1:-1]:
            for margin in grid[:-1]:
                predicted = [
                    row
                    for row in calibration
                    if getattr(row, f"{anatomy}_dice") >= threshold
                    and getattr(row, f"{anatomy}_dice")
                    - getattr(row, f"{other}_dice")
                    >= margin
                ]
                if any(row.expected_class != anatomy for row in predicted):
                    continue
                correct = sum(row.expected_class == anatomy for row in predicted)
                candidate = (correct, threshold, margin)
                if best is None or candidate > best:
                    best = candidate
        if best is None or best[0] == 0:
            raise RuntimeError(f"No zero-error projection policy for {anatomy}")
        min_dice[anatomy] = best[1]
        margins[anatomy] = best[2]

    positives = [row for row in calibration if row.expected_class != "negative"]
    negatives = [row for row in calibration if row.expected_class == "negative"]
    best_negative: tuple[int, float, float] | None = None
    for lung_limit in grid[:-1]:
        for scapula_limit in grid[:-1]:
            if any(
                row.lung_dice <= lung_limit and row.scapula_dice <= scapula_limit
                for row in positives
            ):
                continue
            correct = sum(
                row.lung_dice <= lung_limit and row.scapula_dice <= scapula_limit
                for row in negatives
            )
            candidate = (correct, -lung_limit, -scapula_limit)
            if best_negative is None or candidate > best_negative:
                best_negative = candidate
    if best_negative is None or best_negative[0] == 0:
        raise RuntimeError("No zero-error projection negative policy")
    return {
        "min_dice": min_dice,
        "max_negative_dice": {
            "lung": -best_negative[1],
            "scapula": -best_negative[2],
        },
        "min_class_margin": margins,
    }


def evaluate_projection_policy(
    rows: list[ProjectionBodypartRow],
    split: str,
    policy: dict[str, dict[str, float]],
) -> dict[str, object]:
    selected = [row for row in rows if row.split == split]
    predictions = [classify_projection_candidate(row, policy) for row in selected]
    correct = sum(
        predicted == row.expected_class for row, predicted in zip(selected, predictions)
    )
    wrong = sum(
        predicted is not None and predicted != row.expected_class
        for row, predicted in zip(selected, predictions)
    )
    abstained = sum(predicted is None for predicted in predictions)
    per_class = {}
    for expected in ("lung", "scapula", "negative"):
        indices = [index for index, row in enumerate(selected) if row.expected_class == expected]
        per_class[expected] = {
            "rows": len(indices),
            "correct": sum(predictions[index] == expected for index in indices),
            "wrong_conclusive": sum(
                predictions[index] is not None and predictions[index] != expected
                for index in indices
            ),
            "abstained": sum(predictions[index] is None for index in indices),
        }
    return {
        "split": split,
        "patients": len({row.patient_id for row in selected}),
        "candidate_rows": len(selected),
        "correct_conclusive": correct,
        "wrong_conclusive": wrong,
        "abstained": abstained,
        "conclusive_rate": round((correct + wrong) / len(selected), 6) if selected else 0.0,
        "per_class": per_class,
    }

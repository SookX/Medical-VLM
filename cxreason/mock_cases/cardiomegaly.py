from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from cxreason.integrations.chexstruct import CheXStructRepository


CARDIOMEGALY_STAGE_NAMES = [
    "stage1_criterion",
    "stage2_anatomy",
    "stage3_measurement",
    "stage4_final",
]

CARDIOMEGALY_THRESHOLDS_BY_VIEW = {"PA": 0.495, "AP": 0.545}


@dataclass(frozen=True)
class MockPath1Case:
    case_id: str
    task: str
    source_dataset: str
    stage_answers: dict[str, Any]


def build_cardiomegaly_case(
    source_dataset: str = "nih_cxr14",
    row_index: int = 0,
    chexstruct_root: str = "data/CheXStruct",
) -> MockPath1Case:
    """Create one mock Path-1 cardiomegaly case from CheXStruct CTR fields."""

    repo = CheXStructRepository(chexstruct_root)
    row = repo.load_table(source_dataset, "cardiomegaly").iloc[row_index].to_dict()
    ctr = float(row["ctr"])
    label = bool(int(row["label"]))
    case_id = str(row["image_file"])
    viewposition = str(row["viewposition"]).upper()
    threshold = CARDIOMEGALY_THRESHOLDS_BY_VIEW.get(viewposition, 0.5)
    return MockPath1Case(
        case_id=case_id,
        task="cardiomegaly",
        source_dataset=source_dataset,
        stage_answers={
            "stage1_criterion": "cardiothoracic ratio (CTR)",
            "stage2_anatomy": {
                "required_structures": ["heart transverse width", "thoracic/lung transverse width"],
                "heart_xmin": row["heart_xmin"],
                "heart_xmax": row["heart_xmax"],
                "lung_xmin": row["lung_xmin"],
                "lung_xmax": row["lung_xmax"],
                "viewposition": viewposition,
            },
            "stage3_measurement": {
                "heart_width": row["heart_width"],
                "lung_width": row["lung_width"],
                "ctr": ctr,
            },
            "stage4_final": {
                "cardiomegaly": label,
                "viewposition": viewposition,
                "threshold": threshold,
                "rule": f"cardiomegaly is supported for {viewposition} views when CTR > {threshold}",
            },
        },
    )


class CorruptingCardiomegalyGenerator:
    """Mock generator that can fail specific stages once before repairing them."""

    def __init__(
        self,
        case: MockPath1Case,
        corrupt_stage1_once: bool = False,
        corrupt_stage3_once: bool = True,
        corrupt_stage4_once: bool = False,
    ) -> None:
        self.case = case
        self.corrupt_stage1_once = corrupt_stage1_once
        self.corrupt_stage3_once = corrupt_stage3_once
        self.corrupt_stage4_once = corrupt_stage4_once
        self.calls: dict[str, int] = {}

    def generate(
        self,
        stage: str,
        accepted_outputs: dict[str, Any],
        repair_feedback: str | None = None,
    ) -> Any:
        self.calls[stage] = self.calls.get(stage, 0) + 1
        output = copy.deepcopy(self.case.stage_answers[stage])

        if stage == "stage1_criterion" and self.corrupt_stage1_once and self.calls[stage] == 1:
            return "visual impression of enlarged cardiac silhouette"

        if stage == "stage3_measurement" and self.corrupt_stage3_once and self.calls[stage] == 1:
            output["ctr"] = round(float(output["ctr"]) + 0.2, 4)
            return output

        if stage == "stage4_final" and self.corrupt_stage4_once and self.calls[stage] == 1:
            output["cardiomegaly"] = not bool(output["cardiomegaly"])
            return output

        return output

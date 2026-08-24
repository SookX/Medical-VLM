from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from cxreason.integrations.chexstruct import CheXStructRepository
from cxreason.tasks.base import TaskSpec
from cxreason.tasks.registry import get_task_spec


@dataclass(frozen=True)
class CheXStructMockCase:
    case_id: str
    task: str
    source_dataset: str
    stage_outputs: dict[str, Any]


def _row_value(row: dict[str, Any], field: str) -> Any:
    return row[field]


def build_chexstruct_mock_case(
    task: str,
    source_dataset: str = "nih_cxr14",
    row_index: int = 0,
    chexstruct_root: str = "data/CheXStruct",
) -> CheXStructMockCase:
    spec = get_task_spec(task)
    repo = CheXStructRepository(chexstruct_root)
    row = repo.load_table(source_dataset, task).iloc[row_index].to_dict()
    viewposition = str(row.get("viewposition", "unknown")).upper()

    stage2 = {field: _row_value(row, field) for field in spec.stage2_fields if field in row}
    if "viewposition" in spec.stage2_fields:
        stage2["viewposition"] = viewposition

    stage3 = {field: _row_value(row, field) for field in spec.stage3_fields if field in row}
    final = {
        spec.final_field: bool(int(row["label"])),
        "task": task,
        "viewposition": viewposition,
        "verification_level": "chexstruct_reference_label",
    }

    return CheXStructMockCase(
        case_id=str(row["image_file"]),
        task=task,
        source_dataset=source_dataset,
        stage_outputs={
            "stage1_criterion": {"criterion": spec.preferred_criterion},
            "stage2_anatomy": stage2,
            "stage3_measurement": stage3,
            "stage4_final": final,
        },
    )


class CorruptingCheXStructGenerator:
    """Generic mock generator for task-router development."""

    def __init__(
        self,
        case: CheXStructMockCase,
        spec: TaskSpec | None = None,
        corrupt_stage1_once: bool = False,
        corrupt_stage2_once: bool = False,
        corrupt_stage3_once: bool = True,
        corrupt_stage4_once: bool = False,
    ) -> None:
        self.case = case
        self.spec = spec or get_task_spec(case.task)
        self.corrupt_stage1_once = corrupt_stage1_once
        self.corrupt_stage2_once = corrupt_stage2_once
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
        output = copy.deepcopy(self.case.stage_outputs[stage])

        if stage == "stage1_criterion" and self.corrupt_stage1_once and self.calls[stage] == 1:
            return {"criterion": "general visual impression"}

        if stage == "stage2_anatomy" and self.corrupt_stage2_once and self.calls[stage] == 1:
            if self.spec.stage2_fields:
                output.pop(self.spec.stage2_fields[0], None)
            return output

        if stage == "stage3_measurement" and self.corrupt_stage3_once and self.calls[stage] == 1:
            if self.spec.ratio_checks:
                target = self.spec.ratio_checks[0].target
                output[target] = round(float(output[target]) + 0.2, 4)
            elif self.spec.minmax_ratio_checks:
                target = self.spec.minmax_ratio_checks[0].target
                output[target] = round(float(output[target]) + 0.2, 4)
            elif self.spec.stage3_fields:
                field = self.spec.stage3_fields[0]
                try:
                    output[field] = float(output[field]) + 999
                except (TypeError, ValueError):
                    output.pop(field, None)
            return output

        if stage == "stage4_final" and self.corrupt_stage4_once and self.calls[stage] == 1:
            output[self.spec.final_field] = not bool(output[self.spec.final_field])
            return output

        return output

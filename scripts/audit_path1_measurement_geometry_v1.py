"""Audit exact ratio and Carina-angle geometry on accepted Stage-2 cases."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.data.cxreasonbench import CXReasonBenchPath1Loader
from cxreason.path1.independent_verifiers import extract_red_overlay_mask
from cxreason.path1.measurement_geometry import carina_angle_from_mask, maximum_to_median_width_ratio
from cxreason.path1.practical_verifiers import parse_mcq_options, selected_option_letters
from scripts.run_path1_inclusion_unified import atomic_json, load_json


SPECS = {
    "aortic_knob_enlargement": ("outputs/path1_aortic_knob_stage2_v5_transfer_audit/summary.json", "aortic_knob", "trachea"),
    "descending_aorta_enlargement": ("outputs/path1_descending_aorta_stage2_v2_transfer_audit/summary.json", "descending_aorta", "trachea"),
    "carina_angle": ("outputs/path1_carina_selector_frozen_v2/verification.json", "carina", None),
}


def interval_options(question: str) -> dict[str, tuple[float, float]]:
    result = {}
    for letter, text in parse_mcq_options(question).items():
        values = re.findall(r"-?\d+(?:\.\d+)?", text)
        if len(values) >= 2:
            result[letter] = (float(values[0]), float(values[1]))
    return result


def choose(value: float, options: dict[str, tuple[float, float]]) -> str | None:
    matches = [letter for letter, (low, high) in options.items() if low <= value <= high]
    return matches[0] if len(matches) == 1 else None


def stage2_eligible(task: str, summary: dict) -> set[str]:
    if task != "carina_angle":
        return {row["dicom"] for row in summary["records"] if row["decision"] == "CORRECT_CONCLUSIVE"}
    # The frozen selector verifies six anatomy actions; re-evaluate all cases at
    # Stage 3 because the angle estimator consumes the canonical carina overlay.
    return set()


def main() -> int:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=SPECS)
    args = parser.parse_args()
    task = args.task
    stage2_path, target_name, denominator_name = SPECS[task]
    stage2 = load_json(workspace / stage2_path)
    loader = CXReasonBenchPath1Loader(
        workspace / "physionet.org/files/chexstruct-cxreasonbench/1.0.1/CXReasonBench",
        workspace / "physionet.org/files/mimic-cxr-jpg/2.1.0/files",
    )
    eligible = stage2_eligible(task, stage2)
    if task == "carina_angle":
        eligible = set(loader.case_ids(task))
    records = []
    for dicom in loader.case_ids(task):
        if dicom not in eligible:
            continue
        case = loader.load_case(task, dicom)
        paths = [path for turn in case.anatomy for path in turn.image_paths]
        target_paths = [path for path in paths if path.parent.name == target_name and path.parent.parent.name == task]
        denominator_paths = [] if denominator_name is None else [path for path in paths if path.parent.name == denominator_name and path.parent.parent.name == task]
        try:
            if len(target_paths) != 1 or (denominator_name is not None and len(denominator_paths) != 1):
                raise ValueError("canonical target masks are not unique")
            target = extract_red_overlay_mask(target_paths[0])
            value = carina_angle_from_mask(target) if denominator_name is None else maximum_to_median_width_ratio(target, extract_red_overlay_mask(denominator_paths[0]))
            value = float(round(value)) if task == "carina_angle" else round(value, 2)
            proposed = choose(value, interval_options(case.measurement.question))
            reference = selected_option_letters(case.measurement.question, case.measurement.answer)[0]
            decision = "CORRECT_CONCLUSIVE" if proposed == reference else "ABSTAIN" if proposed is None else "WRONG_CONCLUSIVE"
            records.append({"dicom": dicom, "value": value, "proposed": proposed, "reference": reference, "decision": decision})
        except ValueError as error:
            records.append({"dicom": dicom, "decision": "ABSTAIN", "reason": str(error)})
    correct = sum(row["decision"] == "CORRECT_CONCLUSIVE" for row in records)
    wrong = sum(row["decision"] == "WRONG_CONCLUSIVE" for row in records)
    abstained = sum(row["decision"] == "ABSTAIN" for row in records)
    accepted = wrong == 0 and correct > 0
    output = workspace / f"outputs/path1_{task}_stage3_geometry_v1_transfer_audit"
    report = {"schema_version": 1, "experiment": f"path1_{task}_stage3_geometry_v1_transfer_audit", "status": "passed" if accepted else "rejected", "stage2_eligible": len(eligible), "correct_conclusive": correct, "wrong_conclusive": wrong, "abstained": abstained, "records": records, "claim_boundary": "benchmark transfer/development audit; requires untouched confirmation", "threshold_retuning_allowed": False}
    atomic_json(output / "summary.json", report)
    print(json.dumps({k: report[k] for k in ("status", "stage2_eligible", "correct_conclusive", "wrong_conclusive", "abstained")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

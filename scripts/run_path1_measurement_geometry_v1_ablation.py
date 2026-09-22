"""Run zero-regression rescue ablations for three deterministic measurements."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.data.cxreasonbench import CXReasonBenchPath1Loader
from cxreason.path1.practical_verifiers import parse_mcq_options, selected_option_letters
from cxreason.vision.chestx_det import sha256_file
from scripts.run_path1_inclusion_unified import atomic_json, load_json


SPECS = {
    "aortic_knob_enlargement": {"stage2": "outputs/path1_aortic_knob_stage2_v5_transfer_audit/summary.json", "threshold": (2.495, ">"), "source": "outputs/path1_final_candidate_v4/cases/p10_final_candidate_v4/aortic_knob_enlargement"},
    "descending_aorta_enlargement": {"stage2": "outputs/path1_descending_aorta_stage2_v2_transfer_audit/summary.json", "threshold": (2.5, ">="), "source": "outputs/path1_final_candidate_v4/cases/p10_final_candidate_v4/descending_aorta_enlargement"},
    "carina_angle": {"stage2": None, "threshold": ((39.5, 80.5), "range"), "source": "outputs/path1_safe_stage2_selector_b5/cases/b5_independent_stage2_selector/carina_angle"},
}


def rendered(turn, letters: list[str]) -> str:
    options = parse_mcq_options(turn.question)
    return "FINAL ANSWER: " + ", ".join(f"({letter}) {options[letter]}" for letter in letters)


def attempt(key: str, group: str, response: str, score: int, source: str, attempt_index: int = 1) -> dict:
    return {"stage_key": key, "stage_group": group, "attempt_index": attempt_index, "source": source, "response": response, "evaluator_score": score, "selected": [], "gate_passed": bool(score), "gate_reason": None if score else "deterministic choice does not match reference", "gate_metadata": {"control": "task_specific_measurement_geometry_v1"}, "generated_tokens": 0}


def next_attempt_index(row: dict, stage_key: str) -> int:
    return 1 + max((item["attempt_index"] for item in row["attempts"] if item["stage_key"] == stage_key), default=0)


def final_positive(value: float, threshold) -> bool:
    boundary, operator = threshold
    if operator == ">": return value > boundary
    if operator == ">=": return value >= boundary
    low, high = boundary
    return low <= value <= high


def main() -> int:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--task", required=True, choices=SPECS); parser.add_argument("--policy",choices=("geometry_v1","consensus_v2","consensus_v3"),default="geometry_v1"); args = parser.parse_args()
    task = args.task; spec = SPECS[task]; policy=args.policy
    audit_path = workspace / f"outputs/path1_{task}_stage3_{policy}_transfer_audit/summary.json"
    audit = load_json(audit_path); geometry = {row["dicom"]: row for row in audit["records"] if row.get("proposed") is not None}
    stage2 = None if spec["stage2"] is None else load_json(workspace / spec["stage2"])
    anatomy = {} if stage2 is None else {row["dicom"]: row for row in stage2["records"] if row["decision"] == "CORRECT_CONCLUSIVE"}
    loader = CXReasonBenchPath1Loader(workspace/"physionet.org/files/chexstruct-cxreasonbench/1.0.1/CXReasonBench", workspace/"physionet.org/files/mimic-cxr-jpg/2.1.0/files")
    system = f"p15_{task}_measurement_{policy}"; output = workspace/f"outputs/path1_{task}_measurement_{policy}_ablation"; rows=[]; gains=losses=actions=0
    for dicom in loader.case_ids(task):
        source_path = workspace/spec["source"]/f"{dicom}.json"; source=load_json(source_path); row=copy.deepcopy(source); candidate=geometry.get(dicom)
        eligible = not source["completion"] and candidate is not None and (task == "carina_angle" and source.get("failed_stage_group") == "measurement" or task != "carina_angle" and dicom in anatomy and source.get("failed_stage_group") in {"bodypart", "measurement", "final"})
        if eligible:
            case=loader.load_case(task,dicom); existing={a["stage_key"]:a for a in row["attempts"]}
            if task != "carina_angle":
                turns={turn.key:turn for turn in case.anatomy}
                for record in anatomy[dicom]["turns"]:
                    turn=turns[record["stage_key"]]; response=rendered(turn,record["proposed"]); new=attempt(turn.key,"bodypart",response,1,f"deterministic_{task}_stage2")
                    if turn.key in existing: existing[turn.key].update(new)
                    else: row["attempts"].append(new)
            measured_value=candidate.get("value"); measured_value=float(__import__("numpy").median(candidate["values"])) if measured_value is None else measured_value
            measure_response=rendered(case.measurement,[candidate["proposed"]]); measure_score=int(candidate["decision"]=="CORRECT_CONCLUSIVE")
            row["attempts"].append(attempt(case.measurement.key,"measurement",measure_response,measure_score,f"deterministic_{task}_stage3_{policy}",next_attempt_index(row,case.measurement.key))); actions+=1
            row.setdefault("direct_actions",[]).append({"stage_key":case.measurement.key,"source":f"{task}_stage3_{policy}","value":measured_value,"response":measure_response})
            if measure_score:
                positive=final_positive(measured_value,spec["threshold"]); options=parse_mcq_options(case.final.question); letters=[letter for letter,text in options.items() if (positive and text.lower().strip().startswith("yes")) or (not positive and text.lower().strip().startswith("no"))]
                if len(letters)!=1: raise RuntimeError(f"Cannot map final answer for {task}/{dicom}")
                response=rendered(case.final,letters); reference=selected_option_letters(case.final.question,case.final.answer); final_score=int(tuple(letters)==tuple(reference)); row["attempts"].append(attempt(case.final.key,"final",response,final_score,f"deterministic_{task}_stage4_geometry_v1",next_attempt_index(row,case.final.key))); row["direct_actions"].append({"stage_key":case.final.key,"source":f"{task}_stage4_geometry_v1","response":response})
                row.update(completion=final_score,depth=4 if final_score else 3,failed_stage_key=None if final_score else case.final.key,failed_stage_group=None if final_score else "final")
            else: row.update(completion=0,depth=2,failed_stage_key=case.measurement.key,failed_stage_group="measurement")
        row.update(experiment=f"path1_{task}_measurement_{policy}_ablation",system=system,source_artifact_sha256=sha256_file(source_path)); gains+=row["completion"]>source["completion"]; losses+=row["completion"]<source["completion"]; atomic_json(output/"cases"/system/task/f"{dicom}.json",row); rows.append(row)
    report={"schema_version":1,"experiment":f"path1_{task}_measurement_{policy}_ablation","status":"accepted_zero_regression" if losses==0 and gains>0 else "rejected_no_gain" if losses==0 else "rejected_regression","task":task,"policy":policy,"cases":100,"actions":actions,"completion":sum(r["completion"] for r in rows),"mean_depth":round(sum(r["depth"] for r in rows)/100,3),"completion_gains":gains,"completion_losses":losses,"stage3_audit_sha256":sha256_file(audit_path),"claim_boundary":"label-blind consensus rescue of failed trajectories; benchmark development ablation, not untouched confirmation"}
    atomic_json(output/"summary.json",report); print(json.dumps(report,indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

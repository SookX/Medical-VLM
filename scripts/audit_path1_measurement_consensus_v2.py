"""Label-blind stability gate for three task-specific Stage-3 measurements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.data.cxreasonbench import CXReasonBenchPath1Loader
from cxreason.path1.measurement_geometry import carina_measurement_ensemble, ratio_measurement_ensemble
from cxreason.path1.practical_verifiers import selected_option_letters
from cxreason.vision.chestx_det import sha256_file
from scripts.audit_path1_measurement_geometry_v1 import SPECS, choose, interval_options, stage2_eligible
from scripts.run_path1_inclusion_unified import atomic_json, load_json


MIN_ESTIMATES = 3
FINAL_RULES={"aortic_knob_enlargement":(2.495,">"),"descending_aorta_enlargement":(2.5,">="),"carina_angle":((39.5,80.5),"range")}


def stable_final(interval: tuple[float,float], rule) -> bool:
    low,high=interval; boundary,operator=rule
    if operator in {">",">="}: return high < boundary or low > boundary
    normal_low,normal_high=boundary
    return (low>=normal_low and high<=normal_high) or high<normal_low or low>normal_high


def main() -> int:
    workspace=Path(__file__).resolve().parents[1]; parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--task",required=True,choices=SPECS); parser.add_argument("--version",choices=("v2","v3"),default="v2"); args=parser.parse_args(); task=args.task
    stage2_path,target_name,denominator_name=SPECS[task]; stage2_file=workspace/stage2_path; stage2=load_json(stage2_file)
    loader=CXReasonBenchPath1Loader(workspace/"physionet.org/files/chexstruct-cxreasonbench/1.0.1/CXReasonBench",workspace/"physionet.org/files/mimic-cxr-jpg/2.1.0/files")
    eligible=stage2_eligible(task,stage2) if task!="carina_angle" else set(loader.case_ids(task)); records=[]
    for dicom in loader.case_ids(task):
        if dicom not in eligible: continue
        case=loader.load_case(task,dicom); paths=[p for turn in case.anatomy for p in turn.image_paths]; targets=[p for p in paths if p.parent.name==target_name and p.parent.parent.name==task]; denominators=[] if denominator_name is None else [p for p in paths if p.parent.name==denominator_name and p.parent.parent.name==task]
        options=interval_options(case.measurement.question); reference=selected_option_letters(case.measurement.question,case.measurement.answer)[0]
        if len(targets)!=1 or (denominator_name is not None and len(denominators)!=1): values=[]
        elif denominator_name is None: values=carina_measurement_ensemble(targets[0])
        else: values=ratio_measurement_ensemble(targets[0],denominators[0])
        rounded=[float(round(v)) if task=="carina_angle" else round(float(v),2) for v in values]; letters=[choose(v,options) for v in rounded]; conclusive=len(letters)>=MIN_ESTIMATES and None not in letters and len(set(letters))==1; proposed=letters[0] if conclusive else None; final_stable=proposed is not None and stable_final(options[proposed],FINAL_RULES[task]); proposed=proposed if args.version=="v2" or final_stable else None; decision="CORRECT_CONCLUSIVE" if proposed==reference else "ABSTAIN" if proposed is None else "WRONG_CONCLUSIVE"
        records.append({"dicom":dicom,"values":rounded,"option_votes":letters,"dispersion":round(float(np.ptp(rounded)),6) if rounded else None,"stage4_stable":final_stable,"proposed":proposed,"reference":reference,"decision":decision})
    correct=sum(r["decision"]=="CORRECT_CONCLUSIVE" for r in records); wrong=sum(r["decision"]=="WRONG_CONCLUSIVE" for r in records); abstained=sum(r["decision"]=="ABSTAIN" for r in records); passed=wrong==0 and correct>0
    policy_name=f"consensus_{args.version}"; output=workspace/f"outputs/path1_{task}_stage3_{policy_name}_transfer_audit"; report={"schema_version":1,"experiment":f"path1_{task}_stage3_{policy_name}_transfer_audit","status":"passed" if passed else "rejected","stage2_eligible":len(eligible),"correct_conclusive":correct,"wrong_conclusive":wrong,"abstained":abstained,"policy":{"minimum_estimators":MIN_ESTIMATES,"rule":"all available thresholded largest-component estimates must map to one option","require_stage4_interval_stability":args.version=="v3","reference_answer_used_by_gate":False},"stage2_evidence_sha256":sha256_file(stage2_file),"records":records,"claim_boundary":"label-blind stability rule evaluated on repeatedly inspected benchmark transfer data; requires untouched confirmation","threshold_retuning_allowed":False}; atomic_json(output/"summary.json",report); print(json.dumps({k:report[k] for k in ("status","stage2_eligible","correct_conclusive","wrong_conclusive","abstained","policy")},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

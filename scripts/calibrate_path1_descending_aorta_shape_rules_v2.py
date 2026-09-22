"""Validate conservative descending-aorta/trachea overlay rules externally."""

from __future__ import annotations

import csv,json,sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cxreason.path1.aortic_knob_verifiers import descending_aorta_shape_label
from cxreason.vision.chestx_det import sha256_file
from cxreason.vision.cxas import load_cxas_prediction
from scripts.calibrate_path1_cardiomegaly_stage2 import CANDIDATE_TYPES,candidate_overlay
from scripts.run_path1_inclusion_unified import atomic_json


def expected(t):return t if t in {"descending_aorta","trachea"} else "negative"


def main()->int:
    workspace=Path(__file__).resolve().parents[1];annotations_path=workspace/"data/path1_mediastinal_stage2_calibration/annotations.csv";output=workspace/"outputs/path1_descending_aorta_stage2_v2_calibration";annotations={r["image_file"]:r for r in csv.DictReader(annotations_path.open(encoding="utf-8",newline=""))};paths={}
    for directory in [workspace/"outputs/path1_cardiomegaly_stage2_calibration/cxas_predictions",workspace/"outputs/path1_projection_bodypart_calibration/cxas_predictions"]:
        for path in directory.glob("*.npz"):paths.setdefault(path.stem,path)
    records=defaultdict(list);weights=set()
    for image_file,row in annotations.items():
        if row["split"]=="test" or image_file not in paths:continue
        linked=load_cxas_prediction(paths[image_file]);weights.add(linked.weights_sha256);pred=[]
        for t in CANDIDATE_TYPES:pred.append((t,expected(t),descending_aorta_shape_label(candidate_overlay(t,linked,row))))
        selected={label:[t for t,_,p in pred if p==label] for label in ("descending_aorta","trachea")};conclusive=all(len(v)==1 for v in selected.values());correct=conclusive and all(expected(v[0])==k for k,v in selected.items());records[row["split"]].append({"image_file":image_file,"decision":"correct" if correct else "wrong" if conclusive else "abstain","selected":selected})
    def metrics(rows):return {"cases":len(rows),"correct_conclusive":sum(r["decision"]=="correct" for r in rows),"wrong_conclusive":sum(r["decision"]=="wrong" for r in rows),"abstained":sum(r["decision"]=="abstain" for r in rows)}
    cal=metrics(records["calibration"]);val=metrics(records["validation"]);passed=val["wrong_conclusive"]==0 and val["correct_conclusive"]>0;output.mkdir(parents=True,exist_ok=True);policy={"schema_version":1,"task":"descending_aorta_enlargement","candidate":"scale_robust_overlay_rules_v2","implementation_sha256":sha256_file(workspace/"cxreason/path1/aortic_knob_verifiers.py"),"cxas_weights_sha256":next(iter(weights)),"test_used":False,"benchmark_used":False};atomic_json(output/"locked_policy.json",policy);report={"schema_version":1,"experiment":"path1_descending_aorta_stage2_v2_external_calibration","status":"locked_for_transfer" if passed else "rejected_on_validation","calibration":cal,"validation":val,"deploy_to_transfer_audit":passed,"policy_sha256":sha256_file(output/"locked_policy.json"),"records":dict(records),"test_used":False,"benchmark_used":False};atomic_json(output/"summary.json",report);print(json.dumps({k:report[k] for k in ("status","calibration","validation","deploy_to_transfer_audit","policy_sha256")},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

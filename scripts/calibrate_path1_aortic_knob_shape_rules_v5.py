"""Validate conservative scale-robust aortic-knob/trachea overlay rules externally."""

from __future__ import annotations

import csv, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.path1.aortic_knob_verifiers import scale_robust_shape_label
from cxreason.vision.chestx_det import sha256_file
from cxreason.vision.cxas import load_cxas_prediction
from scripts.calibrate_path1_aortic_knob_stage2_v2 import expected
from scripts.calibrate_path1_cardiomegaly_stage2 import CANDIDATE_TYPES, candidate_overlay
from scripts.run_path1_inclusion_unified import atomic_json


def main() -> int:
    workspace=Path(__file__).resolve().parents[1]; annotations_path=workspace/"data/path1_mediastinal_stage2_calibration/annotations.csv"; output=workspace/"outputs/path1_aortic_knob_stage2_v5_calibration"
    with annotations_path.open("r",encoding="utf-8",newline="") as handle: annotations={r["image_file"]:r for r in csv.DictReader(handle)}
    directories=[workspace/"outputs/path1_cardiomegaly_stage2_calibration/cxas_predictions",workspace/"outputs/path1_projection_bodypart_calibration/cxas_predictions"]
    predictions={}
    for directory in directories:
        for path in directory.glob("*.npz"): predictions.setdefault(path.stem,path)
    by_split=defaultdict(list); weights=set()
    for image_file,row in annotations.items():
        path=predictions.get(image_file)
        if path is None or row["split"]=="test": continue
        linked=load_cxas_prediction(path); weights.add(linked.weights_sha256); labels=[]
        for candidate_type in CANDIDATE_TYPES:
            value=scale_robust_shape_label(candidate_overlay(candidate_type,linked,row)); labels.append((candidate_type,expected(candidate_type),value))
        selected={label:[candidate for candidate,_,prediction in labels if prediction==label] for label in ("aortic_knob","trachea")}
        conclusive=all(len(v)==1 for v in selected.values()); correct=conclusive and all(expected(v[0])==label for label,v in selected.items())
        by_split[row["split"]].append({"image_file":image_file,"decision":"correct" if correct else "wrong" if conclusive else "abstain","selected":selected,"candidate_predictions":labels})
    def metrics(rows): return {"cases":len(rows),"correct_conclusive":sum(r["decision"]=="correct" for r in rows),"wrong_conclusive":sum(r["decision"]=="wrong" for r in rows),"abstained":sum(r["decision"]=="abstain" for r in rows)}
    calibration=metrics(by_split["calibration"]); validation=metrics(by_split["validation"]); passed=validation["wrong_conclusive"]==0 and validation["correct_conclusive"]>0
    output.mkdir(parents=True,exist_ok=True); policy={"schema_version":1,"task":"aortic_knob_enlargement","candidate":"scale_robust_overlay_rules_v5","rule":"scale_robust_shape_label_v1","rule_thresholds":{"aortic_knob":{"aspect":[0.70,1.80],"density_min":0.50,"center_x":[0.42,0.68],"center_y":[0.12,0.35],"bbox_width_max":0.22,"bbox_height_max":0.20},"trachea":{"aspect":[0.28,0.80],"density":[0.15,0.90],"center_x":[0.38,0.62],"center_y":[0.08,0.34],"bbox_height":[0.12,0.55],"bbox_width_max":0.25}},"implementation_sha256":sha256_file(workspace/"cxreason/path1/aortic_knob_verifiers.py"),"cxas_weights_sha256":next(iter(weights)),"selection_split":"external calibration and validation","benchmark_used":False}; atomic_json(output/"locked_policy.json",policy)
    report={"schema_version":1,"experiment":"path1_aortic_knob_stage2_v5_external_calibration","status":"locked_candidate_validation_passed" if passed else "rejected_on_validation","calibration":calibration,"validation":validation,"deploy_to_transfer_audit":passed,"policy_sha256":sha256_file(output/"locked_policy.json"),"records":dict(by_split),"test_used":False,"benchmark_used":False}; atomic_json(output/"summary.json",report); print(json.dumps({k:report[k] for k in ("status","calibration","validation","deploy_to_transfer_audit","policy_sha256")},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

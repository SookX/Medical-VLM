"""Verify the frozen three-task measurement-consensus candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from cxreason.vision.chestx_det import sha256_file
from scripts.run_path1_inclusion_unified import atomic_json,load_json


def require(condition:bool,message:str)->None:
    if not condition: raise RuntimeError(message)


def main()->int:
    workspace=Path(__file__).resolve().parents[1]; config_path=workspace/"configs/path1_three_measurement_consensus_v1.json"; config=load_json(config_path)
    require(config["status"]=="frozen_development_candidate","configuration status drift")
    for relative,expected in config["locks"].items(): require(sha256_file(workspace/relative)==expected,f"hash drift: {relative}")
    expected={"aortic_knob_enlargement":(5,5,0),"carina_angle":(8,0,0),"descending_aorta_enlargement":(23,22,0)}
    paths={"aortic_knob_enlargement":"outputs/path1_aortic_knob_enlargement_measurement_consensus_v3_ablation/summary.json","carina_angle":"outputs/path1_carina_angle_measurement_consensus_v2_ablation/summary.json","descending_aorta_enlargement":"outputs/path1_descending_aorta_enlargement_measurement_consensus_v3_ablation/summary.json"}
    for task,(completion,gains,losses) in expected.items():
        row=load_json(workspace/paths[task]); require((row["completion"],row["completion_gains"],row["completion_losses"])==(completion,gains,losses),f"outcome drift: {task}")
    metrics=load_json(workspace/"outputs/path1_final_candidate_v9_consensus_diagnostic_official/metrics.json"); require(metrics["metrics"]=={"Completion_score":16.75,"Completion_refined":14.0,"Depth_score":1.55,"Depth_refined":1.5,"Consistency_score":41.25,"Alignment_score":42.13,"Alignment_refined":24.83},"official metric drift")
    report={"schema_version":1,"status":"verified_frozen_development_candidate","configuration_sha256":sha256_file(config_path),"tasks":3,"cases":1200,"completion_gains":27,"completion_losses":0,"metrics":metrics["metrics"],"carina_safe_but_inactive":True,"release_result":False}
    output=workspace/"outputs/path1_three_measurement_consensus_v1"; atomic_json(output/"verification.json",report); print(json.dumps(report,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

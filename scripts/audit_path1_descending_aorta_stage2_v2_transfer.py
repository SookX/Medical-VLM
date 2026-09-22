"""Locked transfer audit for descending-aorta/trachea scale-robust rules."""

from __future__ import annotations

import json,sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cxreason.data.cxreasonbench import CXReasonBenchPath1Loader
from cxreason.path1.aortic_knob_verifiers import descending_aorta_shape_label
from cxreason.path1.independent_verifiers import extract_red_overlay_mask
from cxreason.path1.practical_verifiers import _normalize,parse_mcq_options,selected_option_letters
from cxreason.vision.chestx_det import sha256_file
from scripts.run_path1_inclusion_unified import atomic_json,load_json


def main()->int:
    workspace=Path(__file__).resolve().parents[1];cal_dir=workspace/"outputs/path1_descending_aorta_stage2_v2_calibration";output=workspace/"outputs/path1_descending_aorta_stage2_v2_transfer_audit";cal=load_json(cal_dir/"summary.json");policy=load_json(cal_dir/"locked_policy.json")
    if not cal["deploy_to_transfer_audit"] or policy["implementation_sha256"]!=sha256_file(workspace/"cxreason/path1/aortic_knob_verifiers.py"):raise RuntimeError("Calibration gate closed or drifted")
    loader=CXReasonBenchPath1Loader(workspace/"physionet.org/files/chexstruct-cxreasonbench/1.0.1/CXReasonBench",workspace/"physionet.org/files/mimic-cxr-jpg/2.1.0/files");records=[];totals=Counter()
    for dicom in loader.case_ids("descending_aorta_enlargement"):
        case=loader.load_case("descending_aorta_enlargement",dicom);found=set();turns=[];abstain=False
        for turn in case.anatomy:
            labels=[descending_aorta_shape_label(extract_red_overlay_mask(path)) for path in turn.image_paths]
            if any(label in found for label in labels if label in {"descending_aorta","trachea"}):abstain=True;break
            positive={target:[chr(97+i) for i,label in enumerate(labels) if label==target] for target in ("descending_aorta","trachea") if target not in found}
            if any(len(v)>1 for v in positive.values()):abstain=True;break
            selected=[v[0] for v in positive.values() if len(v)==1];found.update(k for k,v in positive.items() if len(v)==1);missing={"descending_aorta","trachea"}-found;special=[l for l,o in parse_mcq_options(turn.question).items() if "need new option" in _normalize(o) or "none of the above" in _normalize(o)]
            if missing and len(special)==1:selected.append(special[0])
            proposed=tuple(sorted(selected));reference=tuple(sorted(selected_option_letters(turn.question,turn.answer)));turns.append({"stage_key":turn.key,"labels":labels,"proposed":list(proposed),"reference":list(reference),"correct":proposed==reference})
        decision="ABSTAIN" if abstain or found!={"descending_aorta","trachea"} else "CORRECT_CONCLUSIVE" if all(t["correct"] for t in turns) else "WRONG_CONCLUSIVE";totals[decision]+=1;records.append({"dicom":dicom,"decision":decision,"turns":turns})
    passed=totals["WRONG_CONCLUSIVE"]==0 and totals["CORRECT_CONCLUSIVE"]>0;report={"schema_version":1,"experiment":"path1_descending_aorta_stage2_v2_transfer","status":"passed" if passed else "rejected","cases":len(records),"correct_conclusive":totals["CORRECT_CONCLUSIVE"],"wrong_conclusive":totals["WRONG_CONCLUSIVE"],"abstained":totals["ABSTAIN"],"records":records,"policy_sha256":sha256_file(cal_dir/"locked_policy.json"),"threshold_retuning_allowed":False};atomic_json(output/"summary.json",report);print(json.dumps({k:report[k] for k in ("status","cases","correct_conclusive","wrong_conclusive","abstained")},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

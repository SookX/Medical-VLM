"""Locked benchmark-format transfer audit for scale-robust aortic overlay rules."""

from __future__ import annotations

import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from cxreason.data.cxreasonbench import CXReasonBenchPath1Loader
from cxreason.path1.aortic_knob_verifiers import scale_robust_shape_label
from cxreason.path1.independent_verifiers import extract_red_overlay_mask
from cxreason.path1.practical_verifiers import parse_mcq_options, selected_option_letters, _normalize
from cxreason.vision.chestx_det import sha256_file
from scripts.run_path1_inclusion_unified import atomic_json,load_json


def main() -> int:
    workspace=Path(__file__).resolve().parents[1]; calibration_dir=workspace/"outputs/path1_aortic_knob_stage2_v5_calibration"; output=workspace/"outputs/path1_aortic_knob_stage2_v5_transfer_audit"
    summary=load_json(calibration_dir/"summary.json")
    if not summary["deploy_to_transfer_audit"]: raise RuntimeError("External rules rejected")
    policy=load_json(calibration_dir/"locked_policy.json")
    if policy["implementation_sha256"]!=sha256_file(workspace/"cxreason/path1/aortic_knob_verifiers.py"): raise RuntimeError("Locked rule implementation drift")
    loader=CXReasonBenchPath1Loader(workspace/"physionet.org/files/chexstruct-cxreasonbench/1.0.1/CXReasonBench",workspace/"physionet.org/files/mimic-cxr-jpg/2.1.0/files"); records=[]; totals=Counter()
    for dicom in loader.case_ids("aortic_knob_enlargement"):
        case=loader.load_case("aortic_knob_enlargement",dicom); found=set(); turn_rows=[]; abstain=False
        for turn in case.anatomy:
            labels=[scale_robust_shape_label(extract_red_overlay_mask(path)) for path in turn.image_paths]
            selected={target:[chr(ord('a')+i) for i,label in enumerate(labels) if label==target and target not in found] for target in ("aortic_knob","trachea")}
            if any(len(v)>1 for v in selected.values()): abstain=True; break
            proposed_values=[v[0] for v in selected.values() if len(v)==1]; found.update(k for k,v in selected.items() if len(v)==1)
            if found!={"aortic_knob","trachea"}:
                special=[letter for letter,text in parse_mcq_options(turn.question).items() if "none of the above" in _normalize(text) or "need new option" in _normalize(text)]
                if len(special)==1: proposed_values.append(special[0])
            proposed=tuple(sorted(proposed_values))
            reference=tuple(sorted(selected_option_letters(turn.question,turn.answer))); correct=proposed==reference
            turn_rows.append({"stage_key":turn.key,"labels":labels,"proposed":list(proposed),"reference":list(reference),"correct":correct})
        decision="ABSTAIN" if abstain or found!={"aortic_knob","trachea"} else "CORRECT_CONCLUSIVE" if all(r["correct"] for r in turn_rows) else "WRONG_CONCLUSIVE"
        totals[decision]+=1; records.append({"dicom":dicom,"decision":decision,"turns":turn_rows})
    passed=totals["WRONG_CONCLUSIVE"]==0 and totals["CORRECT_CONCLUSIVE"]>0; report={"schema_version":1,"experiment":"path1_aortic_knob_stage2_v5_locked_transfer","status":"passed" if passed else "rejected","cases":len(records),"correct_conclusive":totals["CORRECT_CONCLUSIVE"],"wrong_conclusive":totals["WRONG_CONCLUSIVE"],"abstained":totals["ABSTAIN"],"records":records,"policy_sha256":sha256_file(calibration_dir/"locked_policy.json"),"threshold_retuning_allowed":False}; atomic_json(output/"summary.json",report); print(json.dumps({k:report[k] for k in ("status","cases","correct_conclusive","wrong_conclusive","abstained")},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

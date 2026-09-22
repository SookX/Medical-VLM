"""Compare label-blind Carina mask geometry on external annotated splits."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from cxreason.path1.measurement_geometry import carina_angle_extrema, carina_angle_from_mask, carina_angle_pca
from cxreason.vision.cxas import load_cxas_prediction
from scripts.run_path1_inclusion_unified import atomic_json


METHODS={"skeleton":carina_angle_from_mask,"extrema":carina_angle_extrema,"pca":carina_angle_pca}


def main()->int:
    workspace=Path(__file__).resolve().parents[1]; annotations={r["image_file"]:r for r in csv.DictReader((workspace/"data/path1_mediastinal_stage2_calibration/annotations.csv").open(encoding="utf-8",newline=""))}; labels=pd.read_csv(workspace/"data/CheXStruct/nih_cxr14/carina_angle.csv").set_index("image_file")["angle"].to_dict(); predictions={}
    for directory in (workspace/"outputs/path1_cardiomegaly_stage2_calibration/cxas_predictions",workspace/"outputs/path1_projection_bodypart_calibration/cxas_predictions"):
        for path in directory.glob("*.npz"): predictions.setdefault(path.stem,path)
    records=[]
    for image_file,truth in labels.items():
        if image_file not in annotations or image_file not in predictions: continue
        mask=load_cxas_prediction(predictions[image_file]).combined_mask("tracheal bifurcation"); values={}
        for name,method in METHODS.items():
            try: values[name]=method(mask)
            except ValueError: values[name]=None
        records.append({"image_file":image_file,"split":annotations[image_file]["split"],"reference":float(truth),**values})
    metrics={}
    for split in ("calibration","validation","test"):
        rows=[r for r in records if r["split"]==split]; metrics[split]={}
        for name in METHODS:
            valid=[r for r in rows if r[name] is not None]; errors=np.asarray([abs(r[name]-r["reference"]) for r in valid]); metrics[split][name]={"n":len(valid),"mae":round(float(errors.mean()),6),"within_9_5":int((errors<=9.5).sum()),"within_19_5":int((errors<=19.5).sum())}
    output=workspace/"outputs/path1_carina_geometry_external_v1"; report={"schema_version":1,"experiment":"path1_carina_geometry_external_v1","metrics":metrics,"records":records,"benchmark_used":False,"test_used_for_selection":False}; atomic_json(output/"summary.json",report); print(json.dumps(metrics,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

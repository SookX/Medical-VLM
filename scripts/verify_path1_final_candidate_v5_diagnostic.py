"""Verify the 1,200-case v5 diagnostic overlay and its scientific boundary."""

from __future__ import annotations

import argparse, json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.vision.chestx_det import sha256_file
from scripts.run_path1_inclusion_unified import atomic_json, load_json


EXPECTED = {
    "Completion_score": 14.31, "Completion_refined": 13.82,
    "Depth_score": 1.52, "Depth_refined": 1.51,
    "Consistency_score": 48.92, "Alignment_score": 31.11,
    "Alignment_refined": 27.21,
}
EXPECTED_V6 = {
    "Completion_score": 14.61, "Completion_refined": 14.14,
    "Depth_score": 1.53, "Depth_refined": 1.52,
    "Consistency_score": 52.56, "Alignment_score": 31.11,
    "Alignment_refined": 27.21,
}
EXPECTED_V7 = {
    "Completion_score": 20.13, "Completion_refined": 13.88,
    "Depth_score": 1.72, "Depth_refined": 1.57,
    "Consistency_score": 39.13, "Alignment_score": 37.99,
    "Alignment_refined": 22.73,
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> int:
    workspace = Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--version",choices=("v5","v6","v7"),default="v5");args=parser.parse_args();expected={"v5":EXPECTED,"v6":EXPECTED_V6,"v7":EXPECTED_V7}[args.version]
    root = workspace / f"outputs/path1_final_candidate_{args.version}_diagnostic_official"
    metrics_path = root / "metrics.json"; report = load_json(metrics_path)
    require(report["status"] == "official_metric_adapter_complete", "metric adapter incomplete")
    require(report["metrics"] == expected, f"{args.version} diagnostic metric drift")
    require(len(report["replacements"]) == 12, "not all tasks represented")
    require(all(item["cases"] == 100 for item in report["replacements"]), "unequal task coverage")
    require(sum(item["cases"] for item in report["replacements"]) == 1200, "case-count drift")
    gates = {
        "cardiomegaly": ("outputs/path1_cardiomegaly_stage3_v5_ablation/summary.json", "accepted_zero_regression"),
        "inspiration": ("outputs/path1_inspiration_rib_count_rf_v2_ablation/summary.json", "accepted_zero_regression"),
        "ascending_aorta_enlargement": ("outputs/path1_ascending_aorta_v8_ablation/summary.json", "accepted_zero_regression"),
    }
    if args.version in {"v6", "v7"}:
        gates["rotation"] = ("outputs/path1_rotation_v4_ablation/summary.json", "accepted_zero_regression")
    if args.version == "v7":
        for task in ("aortic_knob_enlargement", "carina_angle", "descending_aorta_enlargement"):
            gates[task] = (f"outputs/path1_{task}_measurement_geometry_v1_ablation/summary.json", "accepted_zero_regression")
    evidence = {}
    for task, (relative, status) in gates.items():
        path = workspace / relative; payload = load_json(path)
        require(payload["status"] == status, f"{task} promotion gate drift")
        require(payload.get("completion_losses", 0) == 0, f"{task} has completion regression")
        evidence[task] = {"path": relative, "sha256": sha256_file(path)}
    verification = {
        "schema_version": 1, "status": "verified_diagnostic_not_release",
        "cases": 1200, "tasks": 12, "cases_per_task": 100,
        "metrics": expected, "metrics_sha256": sha256_file(metrics_path),
        "promoted_task_evidence": evidence,
        "all_12_tasks_fully_supported": False,
        "claim_boundary": (
            "Development diagnostic only. Several geometry rules were evaluated after benchmark inspection; "
            "v7 measurement rules are failed-trajectory rescue policies, their standalone transfer audits are imperfect, "
            "and the retired published Gemini scorer is unavailable."
        ),
    }
    atomic_json(root / "verification.json", verification)
    print(json.dumps(verification, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

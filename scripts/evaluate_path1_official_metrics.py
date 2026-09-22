"""Evaluate frozen Path-1 repairs with the original CXReasonBench metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.data.cxreasonbench import CXReasonBenchPath1Loader, PATH1_TASKS
from cxreason.evaluation.official_path1 import (
    atomic_json,
    export_official_layout,
    load_json,
    metric_deltas,
    parse_metric_output,
    run_original_evaluator,
    sha256_file,
)


def parse_replacement(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected TASK=CASE_DIRECTORY")
    task, raw_path = value.split("=", 1)
    if task not in PATH1_TASKS:
        raise argparse.ArgumentTypeError(f"Unknown Path-1 task: {task}")
    if not raw_path:
        raise argparse.ArgumentTypeError("Replacement directory cannot be empty")
    return task, Path(raw_path)


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=workspace
        / "physionet.org/files/chexstruct-cxreasonbench/1.0.1/CXReasonBench",
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=workspace / "physionet.org/files/mimic-cxr-jpg/2.1.0/files",
    )
    parser.add_argument(
        "--baseline-output-dir",
        type=Path,
        default=workspace / "outputs/medgemma4b_path1_reproduction",
    )
    parser.add_argument(
        "--metric-script",
        type=Path,
        default=workspace / "external/CXReasonBench/evaluation/metric.py",
    )
    parser.add_argument(
        "--replacement",
        type=parse_replacement,
        action="append",
        help="Complete replacement cohort as TASK=CASE_DIRECTORY; may be repeated.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Export and score the untouched 1,200-case local baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs/path1_official_metrics_supported_v1",
    )
    parser.add_argument(
        "--label",
        default="medgemma-4b-it+frozen-path1-supported-v1",
    )
    return parser.parse_args()


def default_replacements(workspace: Path) -> dict[str, Path]:
    return {
        "inclusion": workspace
        / "outputs/path1_inclusion_unified/cases/locked_stage2_stage3_stage4/inclusion",
        "projection": workspace
        / (
            "outputs/path1_projection_stage2_strict_confirmation/cases/"
            "selected_stage1_one_stage1_5_two_stage2_strict_two/projection"
        ),
    }


def main() -> int:
    args = parse_args()
    workspace = Path(__file__).resolve().parents[1]
    if args.baseline_only and args.replacement:
        raise ValueError("--baseline-only cannot be combined with --replacement")
    replacements = {} if args.baseline_only else default_replacements(workspace)
    if args.replacement and not args.baseline_only:
        replacements = {}
        for task, path in args.replacement:
            if task in replacements:
                raise ValueError(f"Duplicate replacement task: {task}")
            replacements[task] = path
    missing = [str(path) for path in replacements.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing replacement directories:\n" + "\n".join(missing))
    if not args.metric_script.is_file():
        raise FileNotFoundError(f"Original evaluator not found: {args.metric_script}")

    loader = CXReasonBenchPath1Loader(args.benchmark_root, args.mimic_root)
    manifest = export_official_layout(
        loader=loader,
        baseline_output_dir=args.baseline_output_dir,
        replacements=replacements,
        output_dir=args.output_dir,
        label=args.label,
    )
    inference_root = Path(manifest["inference_root"])
    scoring_root = Path(manifest["scoring_root"])
    metrics, stdout = run_original_evaluator(
        metric_script=args.metric_script,
        inference_root=inference_root,
        scoring_root=scoring_root,
    )
    baseline_metrics_path = args.baseline_output_dir / "metrics.txt"
    baseline_metrics = parse_metric_output(
        baseline_metrics_path.read_text(encoding="utf-8")
    )
    if not replacements and metrics != baseline_metrics:
        raise RuntimeError(
            "Baseline adapter regression failed: "
            f"exported={metrics!r}, expected={baseline_metrics!r}"
        )
    all_task_overlay = set(replacements) == set(PATH1_TASKS)
    scope = (
        "all 1,200 Path-1 cases; untouched local baseline regression"
        if not replacements
        else (
            "all 1,200 Path-1 cases; composed replacements on all 12 tasks"
            if all_task_overlay
            else "all 1,200 Path-1 cases; frozen replacements on supported tasks"
        )
    )
    interpretation = (
        "This is an exact evaluator-layout regression of the local deterministic "
        "baseline; it does not reproduce the retired published scorer."
        if not replacements
        else (
            "This is a full-benchmark all-task deterministic overlay, not an exact "
            "reproduction of the retired published scorer."
            if all_task_overlay
            else (
                "This is a full-benchmark supported-task overlay, not the final all-task "
                "intervention and not an exact reproduction of the retired published scorer."
            )
        )
    )
    result = {
        "schema_version": 1,
        "status": "official_metric_adapter_complete",
        "scope": scope,
        "label": args.label,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "delta_vs_local_baseline": metric_deltas(metrics, baseline_metrics),
        "replacements": manifest["replacements"],
        "evaluator": {
            "path": str(args.metric_script.resolve()),
            "sha256": sha256_file(args.metric_script),
            "scorer": "deterministic_mcq_v1",
            "published_scorer": "gemini-2.0-flash (retired; not used)",
        },
        "interpretation_boundary": interpretation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.txt").write_text(stdout, encoding="utf-8")
    atomic_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

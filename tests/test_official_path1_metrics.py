from pathlib import Path

import pytest

from cxreason.data.cxreasonbench import CXReasonBenchPath1Loader
from cxreason.evaluation.official_path1 import (
    export_case,
    load_json,
    metric_deltas,
    parse_metric_output,
    selected_attempts,
)


WORKSPACE = Path(__file__).resolve().parents[1]
BENCHMARK = (
    WORKSPACE / "physionet.org/files/chexstruct-cxreasonbench/1.0.1/CXReasonBench"
)
MIMIC = WORKSPACE / "physionet.org/files/mimic-cxr-jpg/2.1.0/files"
BASELINE = WORKSPACE / "outputs/medgemma4b_path1_reproduction"
UNIFIED = WORKSPACE / "outputs/path1_inclusion_unified"
requires_local_artifacts = pytest.mark.skipif(
    not (BENCHMARK.exists() and MIMIC.exists() and BASELINE.exists() and UNIFIED.exists()),
    reason="requires local CXReasonBench, MIMIC-CXR, and baseline artifacts",
)


def test_selected_attempts_keeps_accepted_retry_or_terminal_failure() -> None:
    row = {
        "attempts": [
            {
                "stage_key": "init",
                "attempt_index": 1,
                "response": "(a)",
                "evaluator_score": 1,
            },
            {
                "stage_key": "criteria_0",
                "attempt_index": 1,
                "response": "(b)",
                "evaluator_score": 0,
            },
            {
                "stage_key": "criteria_0",
                "attempt_index": 2,
                "response": "(a)",
                "evaluator_score": 1,
            },
        ]
    }
    chosen = selected_attempts(row)
    assert list(chosen) == ["init", "criteria_0"]
    assert chosen["criteria_0"]["attempt_index"] == 2


def test_selected_attempts_accepts_native_not_applicable_score() -> None:
    row = {
        "attempts": [
            {
                "stage_key": "init",
                "attempt_index": 1,
                "response": "(a)",
                "evaluator_score": 1,
            },
            {
                "stage_key": "bodypart_0",
                "attempt_index": 1,
                "response": "(e) None of the above",
                "evaluator_score": -2,
            },
        ]
    }
    assert selected_attempts(row)["bodypart_0"]["evaluator_score"] == -2


@requires_local_artifacts
def test_exported_native_arm_matches_original_scoring_and_responses() -> None:
    loader = CXReasonBenchPath1Loader(BENCHMARK, MIMIC)
    case_path = next(
        iter(sorted((UNIFIED / "cases/b0_native/inclusion").glob("*.json")))
    )
    dicom = case_path.stem
    baseline_inference_path = (
        BASELINE
        / "inference/reasoning/medgemma-4b-it/inclusion"
        / case_path.name
    )
    baseline_scoring_path = (
        BASELINE
        / "scoring/reasoning/deterministic_mcq_v1/medgemma-4b-it/inclusion"
        / case_path.name
    )
    inference, scoring = export_case(
        case=loader.load_case("inclusion", dicom),
        row=load_json(case_path),
        baseline_inference=load_json(baseline_inference_path),
    )
    expected_inference = load_json(baseline_inference_path)
    assert scoring == load_json(baseline_scoring_path)
    for key in scoring:
        if key == "stage-measured_value":
            continue
        assert inference[key]["response"] == expected_inference[key]["response"]


def test_metric_parser_and_deltas_cover_original_table() -> None:
    text = """Path1
Completion_score: 1.73
Completion_refined: 1.32
Depth_score: 0.36
Depth_refined: 0.35
Consistency_score: 32.06
Alignment_score: 43.53
Alignment_refined: 32.74
"""
    metrics = parse_metric_output(text)
    assert metrics["Completion_score"] == 1.73
    assert metric_deltas(metrics, metrics) == {key: 0.0 for key in metrics}


@requires_local_artifacts
def test_export_truncates_downstream_credit_after_incorrect_intermediate_stage() -> None:
    loader = CXReasonBenchPath1Loader(BENCHMARK, MIMIC)
    case_path = next(
        iter(
            sorted(
                (
                    UNIFIED
                    / "cases/locked_stage2_stage3_stage4/inclusion"
                ).glob("*.json")
            )
        )
    )
    row = load_json(case_path)
    row["attempts"][1]["evaluator_score"] = 0
    baseline_path = (
        BASELINE
        / "inference/reasoning/medgemma-4b-it/inclusion"
        / case_path.name
    )
    inference, scoring = export_case(
        case=loader.load_case("inclusion", case_path.stem),
        row=row,
        baseline_inference=load_json(baseline_path),
    )
    assert list(scoring) == ["stage-init", "stage-criteria_0"]
    assert inference["export_source"]["strict_prefix_truncated"] is True

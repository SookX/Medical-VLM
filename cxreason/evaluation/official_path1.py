"""Export controller artifacts to the native CXReasonBench Path-1 layout."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from cxreason.data.cxreasonbench import (
    CXReasonBenchPath1Loader,
    MEASUREMENT_TASKS,
    PATH1_TASKS,
    NativePath1Case,
)


METRIC_KEYS = (
    "Completion_score",
    "Completion_refined",
    "Depth_score",
    "Depth_refined",
    "Consistency_score",
    "Alignment_score",
    "Alignment_refined",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_index_sha256(paths: Iterable[Path]) -> str:
    index = [(path.name, sha256_file(path)) for path in sorted(paths)]
    encoded = json.dumps(index, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_metric_output(text: str) -> dict[str, float | str]:
    """Parse the original evaluator's stable ``Metric_name: value`` output."""

    metrics: dict[str, float | str] = {}
    for key in METRIC_KEYS:
        match = re.search(rf"(?m)^{re.escape(key)}:\s*(\S+)\s*$", text)
        if match is None:
            raise RuntimeError(f"Original evaluator did not emit {key!r}:\n{text}")
        raw = match.group(1)
        metrics[key] = raw if raw == "N/A" else float(raw)
    return metrics


def selected_attempts(row: dict[str, Any]) -> OrderedDict[str, dict[str, Any]]:
    """Return the trajectory attempt retained at each reached stage.

    A successful retry is always the last attempt for its stage because the
    controller stops retrying after acceptance.  If all retries fail, the last
    rejected response is the stage result that stops the trajectory, matching
    the native one-attempt scoring contract.
    """

    attempts = row.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError("Case artifact has no attempts")
    chosen: OrderedDict[str, dict[str, Any]] = OrderedDict()
    last_indices: dict[str, int] = {}
    closed: set[str] = set()
    current_key: str | None = None
    for raw in attempts:
        if not isinstance(raw, dict):
            raise RuntimeError("Case artifact contains a non-object attempt")
        key = raw.get("stage_key")
        index = raw.get("attempt_index")
        if not isinstance(key, str) or not key:
            raise RuntimeError("Attempt is missing stage_key")
        if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
            raise RuntimeError(f"Invalid attempt_index for {key!r}: {index!r}")
        if current_key is not None and key != current_key:
            closed.add(current_key)
        if key in closed:
            raise RuntimeError(f"Non-contiguous attempts for stage {key!r}")
        expected = last_indices.get(key, 0) + 1
        if index != expected:
            raise RuntimeError(
                f"Non-contiguous attempt indices for {key!r}: expected {expected}, got {index}"
            )
        if not isinstance(raw.get("response"), str):
            raise RuntimeError(f"Attempt {key!r}/{index} has no response text")
        score = raw.get("evaluator_score")
        # The deterministic scorer uses -2 for a selected N/A/none option at
        # criteria or anatomy stages. Native baseline scoring files preserve
        # this value, so replacement artifacts must accept it as well.
        if isinstance(score, bool) or score not in (-2, -1, 0, 1):
            raise RuntimeError(
                f"Attempt {key!r}/{index} has invalid evaluator_score {score!r}"
            )
        chosen[key] = raw
        last_indices[key] = index
        current_key = key
    return chosen


def _turn_map(case: NativePath1Case) -> dict[str, Any]:
    return {turn.key: turn for turn in case.all_turns}


def export_case(
    *,
    case: NativePath1Case,
    row: dict[str, Any],
    baseline_inference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one stage-gated artifact into native inference/scoring JSON."""

    if row.get("task") != case.task or row.get("dicom") != case.dicom:
        raise RuntimeError(
            "Artifact identity mismatch: "
            f"expected {case.task}/{case.dicom}, got "
            f"{row.get('task')}/{row.get('dicom')}"
        )
    chosen = selected_attempts(row)
    turns = _turn_map(case)
    expected_keys = [turn.key for turn in case.all_turns]
    observed_keys = list(chosen)
    if observed_keys != expected_keys[: len(observed_keys)]:
        raise RuntimeError(
            f"Artifact is not a native Path-1 prefix for {case.task}/{case.dicom}: "
            f"{observed_keys!r}"
        )

    inference = {
        key: value
        for key, value in baseline_inference.items()
        if not key.startswith("stage-") and key != "scorer_audit"
    }
    inference.update(
        {
            "dicom": case.dicom,
            "cxr_path": case.cxr_path,
            "measured_value": baseline_inference.get("measured_value", ""),
            "scorer": "deterministic_mcq_v1",
            "export_source": {
                "experiment": row.get("experiment"),
                "system": row.get("system"),
            },
        }
    )
    scoring: dict[str, Any] = {}
    scorer_audit: dict[str, Any] = {}
    scored_stage_keys: list[str] = []
    for key, attempt in chosen.items():
        turn = turns[key]
        response = attempt["response"]
        inference[f"stage-{key}"] = {
            "query": turn.question,
            "img_path": [str(path) for path in turn.image_paths],
            "response": response,
            "answer": turn.answer,
        }
        scoring[f"stage-{key}"] = int(attempt["evaluator_score"])
        scorer_audit[key] = {
            "score": int(attempt["evaluator_score"]),
            "selected": attempt.get("selected", []),
            "method": "exported_deterministic_choice_score",
            "attempt_index": int(attempt["attempt_index"]),
            "gate_passed": bool(attempt.get("gate_passed", False)),
        }
        scored_stage_keys.append(key)
        score = int(attempt["evaluator_score"])
        # The paper's runner enters Path 1 for any non-IDK initial response,
        # then advances only across correct intermediate stages. Independent
        # verifier routing can continue across a benchmark-incorrect stage;
        # those downstream turns must not receive credit in the paper metric.
        if (key == "init" and score == -1) or (
            key not in {"init", "final"} and score != 1
        ):
            break
    if case.task in MEASUREMENT_TASKS and "final" in scored_stage_keys:
        scoring["stage-measured_value"] = "deterministic"
    inference["export_source"].update(
        {
            "controller_stage_keys": list(chosen),
            "official_scored_stage_keys": scored_stage_keys,
            "strict_prefix_truncated": len(scored_stage_keys) < len(chosen),
        }
    )
    inference["scorer_audit"] = scorer_audit
    return inference, scoring


def _copy_baseline_tree(source: Path, destination: Path) -> int:
    count = 0
    for task in PATH1_TASKS:
        source_task = source / task
        if not source_task.is_dir():
            raise FileNotFoundError(f"Missing baseline task directory: {source_task}")
        destination_task = destination / task
        destination_task.mkdir(parents=True, exist_ok=True)
        for path in sorted(source_task.glob("*.json")):
            shutil.copy2(path, destination_task / path.name)
            count += 1
    return count


def export_official_layout(
    *,
    loader: CXReasonBenchPath1Loader,
    baseline_output_dir: Path,
    replacements: dict[str, Path],
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    """Overlay complete task trajectories onto the 1,200-case baseline."""

    baseline_inference_root = (
        baseline_output_dir / "inference/reasoning/medgemma-4b-it"
    )
    baseline_scoring_root = (
        baseline_output_dir
        / "scoring/reasoning/deterministic_mcq_v1/medgemma-4b-it"
    )
    config_path = baseline_inference_root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing baseline config: {config_path}")
    unknown = set(replacements) - set(PATH1_TASKS)
    if unknown:
        raise ValueError(f"Unknown replacement tasks: {sorted(unknown)}")

    inference_root = output_dir / "inference/reasoning/medgemma-4b-it"
    scoring_root = (
        output_dir / "scoring/reasoning/deterministic_mcq_v1/medgemma-4b-it"
    )
    inference_root.mkdir(parents=True, exist_ok=True)
    scoring_root.mkdir(parents=True, exist_ok=True)
    copied_inference = _copy_baseline_tree(baseline_inference_root, inference_root)
    copied_scoring = _copy_baseline_tree(baseline_scoring_root, scoring_root)
    if copied_inference != 1200 or copied_scoring != 1200:
        raise RuntimeError(
            "Baseline must contain exactly 1,200 inference and scoring files; "
            f"found {copied_inference} and {copied_scoring}"
        )

    replacement_records: list[dict[str, Any]] = []
    for task, case_dir in sorted(replacements.items()):
        paths = sorted(case_dir.glob("*.json"))
        expected_dicoms = set(loader.case_ids(task))
        observed_dicoms = {path.stem for path in paths}
        if len(paths) != 100 or observed_dicoms != expected_dicoms:
            missing = sorted(expected_dicoms - observed_dicoms)
            extra = sorted(observed_dicoms - expected_dicoms)
            raise RuntimeError(
                f"Replacement {task!r} must contain the exact 100-case cohort; "
                f"found {len(paths)}, missing={missing[:3]}, extra={extra[:3]}"
            )
        strict_prefix_truncations = 0
        downstream_controller_stages_not_scored = 0
        for path in paths:
            dicom = path.stem
            baseline_path = baseline_inference_root / task / path.name
            inference, scoring = export_case(
                case=loader.load_case(task, dicom),
                row=load_json(path),
                baseline_inference=load_json(baseline_path),
            )
            atomic_json(inference_root / task / path.name, inference)
            atomic_json(scoring_root / task / path.name, scoring)
            source = inference["export_source"]
            controller_keys = source["controller_stage_keys"]
            scored_keys = source["official_scored_stage_keys"]
            strict_prefix_truncations += int(source["strict_prefix_truncated"])
            downstream_controller_stages_not_scored += len(controller_keys) - len(
                scored_keys
            )
        replacement_records.append(
            {
                "task": task,
                "case_directory": str(case_dir.resolve()),
                "cases": len(paths),
                "canonical_artifact_index_sha256": canonical_index_sha256(paths),
                "strict_prefix_truncations": strict_prefix_truncations,
                "downstream_controller_stages_not_scored": (
                    downstream_controller_stages_not_scored
                ),
            }
        )

    config = load_json(config_path)
    config.update(
        {
            "model_id": label,
            "evaluation_path": "reasoning",
            "qa_base_dir": loader.qa_root.as_posix(),
            "official_layout_export": {
                "schema_version": 1,
                "baseline_output_dir": str(baseline_output_dir.resolve()),
                "replacements": replacement_records,
                "case_policy": (
                    "full 1,200-case baseline with complete 100-case task overlays"
                ),
            },
        }
    )
    atomic_json(inference_root / "config.json", config)
    manifest = {
        "schema_version": 1,
        "label": label,
        "cases": 1200,
        "baseline_cases": 1200 - 100 * len(replacements),
        "replacement_cases": 100 * len(replacements),
        "replacements": replacement_records,
        "inference_root": str(inference_root.resolve()),
        "scoring_root": str(scoring_root.resolve()),
    }
    atomic_json(output_dir / "export_manifest.json", manifest)
    return manifest


def run_original_evaluator(
    *, metric_script: Path, inference_root: Path, scoring_root: Path
) -> tuple[dict[str, float | str], str]:
    command = [
        sys.executable,
        str(metric_script),
        "--saved_dir_inference",
        str(inference_root),
        "--saved_dir_scoring",
        str(scoring_root),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            "Original CXReasonBench evaluator failed:\n"
            + completed.stdout
            + completed.stderr
        )
    return parse_metric_output(completed.stdout), completed.stdout


def metric_deltas(
    candidate: dict[str, float | str], baseline: dict[str, float | str]
) -> dict[str, float | str]:
    deltas: dict[str, float | str] = {}
    for key in METRIC_KEYS:
        left, right = candidate[key], baseline[key]
        deltas[key] = (
            round(float(left) - float(right), 4)
            if isinstance(left, float) and isinstance(right, float)
            else "N/A"
        )
    return deltas

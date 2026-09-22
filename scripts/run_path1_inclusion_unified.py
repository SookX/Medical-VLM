"""Run the locked cumulative and unified inclusion policies on all 100 cases."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.data.cxreasonbench import CXReasonBenchPath1Loader
from cxreason.generators.native_medgemma import NativeMedGemmaGenerator
from cxreason.modeling.medgemma_session import MedGemmaSession
from cxreason.path1.execution import NativePath1Executor
from cxreason.path1.independent_verifiers import (
    IndependentInclusionBodypartVerifier,
    IndependentInclusionMeasurementVerifier,
)
from cxreason.path1.practical_verifiers import PracticalFinalConsistencyVerifier
from cxreason.path1.support_policy import load_task_support_policy
from cxreason.vision.chestx_det import load_prediction, sha256_file
from scripts.run_path1_oracle_end_to_end_pilot import (
    CachedEndToEndGenerator,
    core_depth,
)


EXPERIMENT = "path1_inclusion_unified_v1"
SYSTEMS = (
    {"name": "b0_native", "repair_stages": []},
    {"name": "locked_stage2", "repair_stages": ["stage2"]},
    {"name": "locked_stage2_stage3", "repair_stages": ["stage2", "stage3"]},
    {
        "name": "locked_stage2_stage3_stage4",
        "repair_stages": ["stage2", "stage3", "stage4"],
    },
)
STAGE_GROUPS = ("bodypart", "measurement", "final")


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
        "--stage2-policy",
        type=Path,
        default=workspace / "outputs/path1_independent_lung_calibration/locked_policy.json",
    )
    parser.add_argument(
        "--stage2-audit",
        type=Path,
        default=workspace / "outputs/path1_locked_independent_inclusion_audit/summary.json",
    )
    parser.add_argument(
        "--stage3-policy",
        type=Path,
        default=workspace / "outputs/path1_independent_stage3_calibration/locked_policy.json",
    )
    parser.add_argument(
        "--stage3-audit",
        type=Path,
        default=workspace / "outputs/path1_locked_independent_stage3_audit/summary.json",
    )
    parser.add_argument(
        "--stage4-audit",
        type=Path,
        default=workspace / "outputs/path1_practical_final_verifier_audit/summary.json",
    )
    parser.add_argument(
        "--segmenter-audit-dir",
        type=Path,
        default=workspace / "outputs/path1_independent_inclusion_segmenter_audit",
    )
    parser.add_argument(
        "--support-policy",
        type=Path,
        default=workspace / "configs/path1_task_support_v1.json",
    )
    parser.add_argument(
        "--shared-generation-cache",
        type=Path,
        action="append",
        default=None,
        help="Cache directory to merge; may be repeated.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            r"C:\Users\Vasil\.cache\huggingface\hub\models--google--medgemma-4b-it"
            r"\snapshots\290cda5eeccbee130f987c4ad74a59ae6f196408"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs/path1_inclusion_unified",
    )
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def result_path(output_dir: Path, system: str, dicom: str) -> Path:
    return output_dir / "cases" / system / "inclusion" / f"{dicom}.json"


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def merge_caches(sources: list[Path], destination: Path) -> list[dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    provenance: list[dict[str, Any]] = []
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(f"Generation cache not found: {source}")
        source_files = sorted(source.glob("*.json"))
        for path in source_files:
            target = destination / path.name
            if target.is_file():
                if sha256_file(target) != sha256_file(path):
                    raise RuntimeError(f"Generation cache collision: {path.name}")
            else:
                shutil.copy2(path, target)
        provenance.append(
            {
                "source": str(source),
                "source_files": len(source_files),
                "canonical_index_sha256": canonical_sha256(
                    [(path.name, sha256_file(path)) for path in source_files]
                ),
            }
        )
    return provenance


def exact_sign_test(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _comparison(
    reference: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    wins = sum(row["depth"] > reference[row["dicom"]]["depth"] for row in rows)
    losses = sum(row["depth"] < reference[row["dicom"]]["depth"] for row in rows)
    completion_gains = sum(
        row["completion"] > reference[row["dicom"]]["completion"] for row in rows
    )
    completion_losses = sum(
        row["completion"] < reference[row["dicom"]]["completion"] for row in rows
    )
    return {
        "depth_wins": wins,
        "depth_losses": losses,
        "depth_ties": len(rows) - wins - losses,
        "paired_depth_sign_test_p": round(exact_sign_test(wins, losses), 8),
        "completion_gains": completion_gains,
        "completion_losses": completion_losses,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_system = {
        system["name"]: {
            row["dicom"]: row
            for row in results
            if row["system"] == system["name"]
        }
        for system in SYSTEMS
    }
    baseline = by_system["b0_native"]
    systems: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    previous_name = "b0_native"
    for system in SYSTEMS:
        name = system["name"]
        rows = list(by_system[name].values())
        if not rows:
            continue
        repairs = [
            attempt
            for row in rows
            for attempt in row["attempts"]
            if attempt["attempt_index"] > 1
        ]
        repair_calls = Counter(attempt["stage_group"] for attempt in repairs)
        repair_correct = Counter(
            attempt["stage_group"]
            for attempt in repairs
            if attempt["evaluator_score"] == 1
        )
        repair_accepted = Counter(
            attempt["stage_group"] for attempt in repairs if attempt["gate_passed"]
        )
        initial_decisions: dict[str, Counter[str]] = {
            group: Counter() for group in STAGE_GROUPS
        }
        for row in rows:
            for attempt in row["attempts"]:
                if attempt["attempt_index"] != 1:
                    continue
                decision = (attempt.get("gate_metadata") or {}).get("decision")
                if attempt["stage_group"] in initial_decisions and decision:
                    initial_decisions[attempt["stage_group"]][str(decision)] += 1
        completion = sum(row["completion"] for row in rows)
        mean_depth = sum(row["depth"] for row in rows) / len(rows)
        baseline_completion = sum(
            baseline[row["dicom"]]["completion"] for row in rows
        )
        baseline_depth = sum(baseline[row["dicom"]]["depth"] for row in rows) / len(
            rows
        )
        systems.append(
            {
                "system": name,
                "repair_stages": system["repair_stages"],
                "cases": len(rows),
                "completion": completion,
                "completion_percent": round(100 * completion / len(rows), 2),
                "completion_delta_pp": round(
                    100 * (completion - baseline_completion) / len(rows), 2
                ),
                "mean_depth": round(mean_depth, 3),
                "mean_depth_delta": round(mean_depth - baseline_depth, 3),
                "versus_native": _comparison(baseline, rows),
                "repair_calls_by_group": {
                    group: repair_calls[group] for group in STAGE_GROUPS
                },
                "correct_repair_calls_by_group": {
                    group: repair_correct[group] for group in STAGE_GROUPS
                },
                "accepted_repair_calls_by_group": {
                    group: repair_accepted[group] for group in STAGE_GROUPS
                },
                "initial_gate_decisions": {
                    group: dict(sorted(counts.items()))
                    for group, counts in initial_decisions.items()
                },
                "logical_model_calls": sum(row["model_calls_logical"] for row in rows),
                "logical_generated_tokens": sum(
                    row["generated_tokens_logical"] for row in rows
                ),
                "live_model_calls": sum(row["live_model_calls"] for row in rows),
                "generation_cache_hits": sum(
                    row["generation_cache_hits"] for row in rows
                ),
            }
        )
        if name != "b0_native":
            previous = by_system[previous_name]
            transitions.append(
                {
                    "from": previous_name,
                    "to": name,
                    **_comparison(previous, rows),
                }
            )
        previous_name = name

    complete = len(systems) == len(SYSTEMS) and all(
        system["cases"] == 100 for system in systems
    )
    full = next(
        (row for row in systems if row["system"] == SYSTEMS[-1]["name"]), None
    )
    ready = bool(
        complete
        and full
        and full["mean_depth_delta"] > 0
        and full["completion_delta_pp"] >= 0
        and full["versus_native"]["depth_losses"] == 0
        and full["versus_native"]["completion_losses"] == 0
        and all(row["depth_losses"] == 0 for row in transitions)
        and all(row["completion_losses"] == 0 for row in transitions)
    )
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "ready_to_freeze" if ready else "freeze_gate_failed",
        "cohort": "all 100 frozen inclusion cases",
        "systems": systems,
        "cumulative_transitions": transitions,
        "freeze_acceptance": {
            "all_four_arms_have_100_cases": complete,
            "full_policy_improves_mean_depth": bool(full and full["mean_depth_delta"] > 0),
            "full_policy_has_no_depth_regressions": bool(
                full and full["versus_native"]["depth_losses"] == 0
            ),
            "full_policy_has_no_completion_regressions": bool(
                full and full["versus_native"]["completion_losses"] == 0
            ),
            "each_added_stage_has_no_depth_regressions": bool(
                transitions and all(row["depth_losses"] == 0 for row in transitions)
            ),
            "each_added_stage_has_no_completion_regressions": bool(
                transitions and all(row["completion_losses"] == 0 for row in transitions)
            ),
            "ready_to_freeze": ready,
        },
    }


def _serialize(case, state, generator, system: dict[str, Any], dicom: str) -> dict[str, Any]:
    depth = core_depth(case, state)
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "system": system["name"],
        "repair_stages": system["repair_stages"],
        "task": "inclusion",
        "dicom": dicom,
        "depth": depth,
        "completion": int(depth == 4),
        "failed_stage_key": state.failed_stage_key,
        "failed_stage_group": state.failed_stage_group,
        "model_calls_logical": state.model_calls,
        "generated_tokens_logical": state.generated_tokens,
        "verifier_calls": state.verifier_calls,
        "baseline_replays": generator.baseline_replays,
        "generation_cache_hits": generator.generation_cache_hits,
        "live_model_calls": generator.live_calls,
        "attempts": [
            {
                "stage_key": attempt.stage_key,
                "stage_group": attempt.stage_group,
                "attempt_index": attempt.attempt_index,
                "source": generator.sources[index],
                "response": attempt.response,
                "evaluator_score": attempt.evaluator.score,
                "selected": list(attempt.evaluator.selected),
                "gate_passed": attempt.gate_result.passed,
                "gate_reason": attempt.gate_result.reason,
                "gate_metadata": attempt.gate_result.metadata,
                "repair_enabled": attempt.repair_enabled,
                "verifier": attempt.verifier_name,
                "generated_tokens": attempt.generated_tokens,
            }
            for index, attempt in enumerate(state.attempts)
        ],
    }


def main() -> int:
    args = parse_args()
    workspace = Path(__file__).resolve().parents[1]
    support = load_task_support_policy(args.support_policy)
    for system in SYSTEMS:
        support.validate_requested_stages("inclusion", system["repair_stages"])

    stage2_policy = load_json(args.stage2_policy)
    stage3_policy = load_json(args.stage3_policy)
    stage2_audit = load_json(args.stage2_audit)
    stage3_audit = load_json(args.stage3_audit)
    stage4_audit = load_json(args.stage4_audit)
    if stage2_audit["locked_policy_sha256"] != sha256_file(args.stage2_policy):
        raise RuntimeError("Stage-2 audit/policy hash mismatch")
    if stage3_audit["locked_policy_sha256"] != sha256_file(args.stage3_policy):
        raise RuntimeError("Stage-3 audit/policy hash mismatch")
    if (
        stage2_audit["mapping_audit"]["wrong"] != 0
        or stage2_audit["frozen_medgemma_stage2_cohort"][
            "correct_flagged_for_repair"
        ]
        != 0
    ):
        raise RuntimeError("Stage-2 deployment gate is closed")
    if not stage3_audit["deployment_decision"]["run_repair_ablation"]:
        raise RuntimeError("Stage-3 deployment gate is closed")
    if stage4_audit["verifier"] != PracticalFinalConsistencyVerifier.name:
        raise RuntimeError("Unexpected Stage-4 audit verifier")

    inference_root = (
        args.baseline_output_dir / "inference/reasoning/medgemma-4b-it/inclusion"
    )
    dicoms = [path.stem for path in sorted(inference_root.glob("*.json"))]
    if len(dicoms) != 100 or len(set(dicoms)) != 100:
        raise RuntimeError(f"Expected 100 unique inclusion baselines, found {len(dicoms)}")
    mask_root = args.segmenter_audit_dir / "masks"
    missing_masks = [dicom for dicom in dicoms if not (mask_root / f"{dicom}.npz").is_file()]
    if missing_masks:
        raise FileNotFoundError(f"Missing independent masks: {missing_masks[:3]}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Local MedGemma snapshot not found: {args.model_path}")

    jobs = [(system, dicom) for system in SYSTEMS for dicom in dicoms]
    pending = [
        (system, dicom)
        for system, dicom in jobs
        if not (args.resume and result_path(args.output_dir, system["name"], dicom).is_file())
    ]
    print(f"Unified inclusion run: {len(jobs)} jobs; {len(pending)} pending", flush=True)
    if args.dry_run:
        return 0

    default_caches = [
        workspace / "outputs/path1_locked_independent_stage2_ablation/generation_cache",
        workspace / "outputs/path1_locked_independent_stage3_ablation/generation_cache",
        workspace / "outputs/path1_practical_final_pilot/generation_cache",
        workspace / "outputs/path1_practical_inclusion_pilot/generation_cache",
    ]
    cache_sources = args.shared_generation_cache or default_caches
    cache_provenance = merge_caches(
        cache_sources, args.output_dir / "generation_cache"
    )
    run_config = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "task": "inclusion",
        "cases": 100,
        "dicoms": dicoms,
        "systems": list(SYSTEMS),
        "support_policy": str(args.support_policy),
        "support_policy_sha256": sha256_file(args.support_policy),
        "model_id": "google/medgemma-4b-it",
        "model_revision": args.model_path.name,
        "img_size": args.img_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "decoding": "greedy",
        "max_attempts_per_stage": 2,
        "feedback": "stage-specific generic answer-free defaults",
        "accepted_history": "full accepted native response; rejected attempts excluded",
        "stage2": {
            "verifier": IndependentInclusionBodypartVerifier.name,
            "policy": str(args.stage2_policy),
            "policy_sha256": sha256_file(args.stage2_policy),
            "audit": str(args.stage2_audit),
            "audit_sha256": sha256_file(args.stage2_audit),
            "thresholds": {
                key: stage2_policy[key]
                for key in ("min_dice", "max_special_dice", "min_dice_margin")
            },
        },
        "stage3": {
            "verifier": IndependentInclusionMeasurementVerifier.name,
            "policy": str(args.stage3_policy),
            "policy_sha256": sha256_file(args.stage3_policy),
            "audit": str(args.stage3_audit),
            "audit_sha256": sha256_file(args.stage3_audit),
            "thresholds": stage3_policy["min_visible_margins"],
        },
        "stage4": {
            "verifier": PracticalFinalConsistencyVerifier.name,
            "audit": str(args.stage4_audit),
            "audit_sha256": sha256_file(args.stage4_audit),
        },
        "independent_masks": {
            "directory": str(mask_root),
            "count": 100,
        },
        "shared_generation_caches": cache_provenance,
    }
    atomic_json(args.output_dir / "run_config.json", run_config)

    results = [
        load_json(result_path(args.output_dir, system["name"], dicom))
        for system, dicom in jobs
        if result_path(args.output_dir, system["name"], dicom).is_file()
    ]
    if not pending:
        summary = summarize(results)
        old_path = args.output_dir / "summary.json"
        old = load_json(old_path) if old_path.is_file() else {}
        for key in ("wall_seconds", "model_load_seconds", "peak_gpu_memory_gib"):
            if key in old:
                summary[key] = old[key]
        atomic_json(old_path, summary)
        print(json.dumps(summary, indent=2))
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loader = CXReasonBenchPath1Loader(args.benchmark_root, args.mimic_root)
    cases = {dicom: loader.load_case("inclusion", dicom) for dicom in dicoms}
    print("Loading local MedGemma 4B...", flush=True)
    load_started = time.perf_counter()
    session = MedGemmaSession(args.model_path, args.img_size, args.max_new_tokens)
    live_generator = NativeMedGemmaGenerator(session)
    model_load_seconds = time.perf_counter() - load_started
    print(f"Model loaded in {model_load_seconds:.1f}s", flush=True)
    cache_context = {
        "model_revision": args.model_path.name,
        "img_size": args.img_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "decoding": "greedy",
    }
    stage2_thresholds = run_config["stage2"]["thresholds"]
    stage3_thresholds = run_config["stage3"]["thresholds"]
    started = time.perf_counter()
    for index, (system, dicom) in enumerate(pending, start=1):
        case = cases[dicom]
        generator = CachedEndToEndGenerator(
            saved_inference=load_json(inference_root / f"{dicom}.json"),
            live_generator=live_generator,
            cache_dir=args.output_dir / "generation_cache",
            cache_context=cache_context,
        )
        prediction = None
        verifiers = {}
        if "stage2" in system["repair_stages"] or "stage3" in system["repair_stages"]:
            prediction = load_prediction(mask_root / f"{dicom}.npz")
        if "stage2" in system["repair_stages"]:
            verifiers["bodypart"] = IndependentInclusionBodypartVerifier(
                prediction=prediction, **stage2_thresholds
            )
        if "stage3" in system["repair_stages"]:
            verifiers["measurement"] = IndependentInclusionMeasurementVerifier(
                prediction=prediction,
                measurement_question=case.measurement.question,
                min_visible_margins=stage3_thresholds,
            )
        if "stage4" in system["repair_stages"]:
            verifiers["final"] = PracticalFinalConsistencyVerifier.from_case(case)
        state = NativePath1Executor(
            repair_stages=system["repair_stages"],
            verifiers=verifiers,
            max_attempts_per_stage=2,
        ).run(case, generator)
        row = _serialize(case, state, generator, system, dicom)
        atomic_json(result_path(args.output_dir, system["name"], dicom), row)
        results.append(row)
        retry_groups = [
            attempt["stage_group"]
            for attempt in row["attempts"]
            if attempt["attempt_index"] > 1
        ]
        print(
            f"[{index}/{len(pending)}] {system['name']} {dicom} "
            f"depth={row['depth']} retries={retry_groups} "
            f"live={generator.live_calls} cache={generator.generation_cache_hits}",
            flush=True,
        )

    summary = summarize(results)
    summary["wall_seconds"] = round(time.perf_counter() - started, 3)
    summary["model_load_seconds"] = round(model_load_seconds, 3)
    summary["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    del live_generator, session
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the CXReasonBench Path-1 MedGemma-4B greedy baseline locally.

This runner preserves the official multi-stage routing, prompt text, image
resizing, chat history, and output layout. It replaces the retired Gemini 2.0
Flash answer-equivalence grader with a deterministic multiple-choice scorer.
Consequently, it is a close offline reproduction, not a claim that the exact
published row can be recovered with the historical grader unavailable.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.data.cxreasonbench import (
    CXReasonBenchPath1Loader,
    MEASUREMENT_TASKS,
    PATH1_TASKS,
    NativeStageTurn,
)
from cxreason.evaluation.choice_scorer import ChoiceScore, score_choice
from cxreason.modeling.medgemma_session import SYSTEM_MESSAGE, MedGemmaSession


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    default_benchmark = (
        workspace
        / "physionet.org/files/chexstruct-cxreasonbench/1.0.1/CXReasonBench"
    )
    default_mimic = workspace / "physionet.org/files/mimic-cxr-jpg/2.1.0/files"
    default_model = Path(
        r"C:\Users\Vasil\.cache\huggingface\hub\models--google--medgemma-4b-it"
        r"\snapshots\290cda5eeccbee130f987c4ad74a59ae6f196408"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=default_benchmark)
    parser.add_argument("--mimic-root", type=Path, default=default_mimic)
    parser.add_argument("--model-path", type=Path, default=default_model)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "outputs/medgemma4b_path1_reproduction",
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=PATH1_TASKS, default=list(PATH1_TASKS)
    )
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--estimate-seconds-per-case",
        type=float,
        default=10.56,
        help="Initial ETA prior measured on this RTX 3080; live timings replace it after case 1.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-upstream-metrics", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def human_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def add_stage(
    *,
    session: MedGemmaSession,
    history: list[tuple[str, list[Path], str]],
    inference: dict[str, Any],
    scoring: dict[str, Any],
    scorer_audit: dict[str, Any],
    turn: NativeStageTurn,
) -> int:
    image_paths = list(turn.image_paths)
    response = session.generate(
        query=turn.question, image_paths=image_paths, history=history
    )
    history.append((turn.question, image_paths, response))
    result = score_choice(
        stage=turn.scorer_stage,
        question=turn.question,
        answer=turn.answer,
        response=response,
    )
    inference[f"stage-{turn.key}"] = {
        "query": turn.question,
        "img_path": [str(path) for path in image_paths],
        "response": response,
        "answer": turn.answer,
    }
    scoring[f"stage-{turn.key}"] = result.score
    scorer_audit[turn.key] = asdict(result)
    return result.score


def run_case(
    *,
    session: MedGemmaSession,
    loader: CXReasonBenchPath1Loader,
    task: str,
    dicom: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case = loader.load_case(task, dicom)
    history: list[tuple[str, list[Path], str]] = []
    scoring: dict[str, Any] = {}
    scorer_audit: dict[str, Any] = {}
    inference: dict[str, Any] = {
        "dicom": dicom,
        "system_message": SYSTEM_MESSAGE,
        "cxr_path": case.cxr_path,
        "measured_value": "",
        "scorer": "deterministic_mcq_v1",
    }
    score_init = add_stage(
        session=session,
        history=history,
        inference=inference,
        scoring=scoring,
        scorer_audit=scorer_audit,
        turn=case.initial,
    )

    if score_init != -1:
        criteria_scores: list[int] = []
        for turn in case.criteria:
            score = add_stage(
                session=session,
                history=history,
                inference=inference,
                scoring=scoring,
                scorer_audit=scorer_audit,
                turn=turn,
            )
            criteria_scores.append(score)
            if score != 1:
                break

        if criteria_scores and len(criteria_scores) == sum(criteria_scores):
            if case.criterion_refinement is not None:
                custom_score = add_stage(
                    session=session,
                    history=history,
                    inference=inference,
                    scoring=scoring,
                    scorer_audit=scorer_audit,
                    turn=case.criterion_refinement,
                )
            else:
                custom_score = 1

            if custom_score == 1:
                body_scores: list[int] = []
                for turn in case.anatomy:
                    score = add_stage(
                        session=session,
                        history=history,
                        inference=inference,
                        scoring=scoring,
                        scorer_audit=scorer_audit,
                        turn=turn,
                    )
                    body_scores.append(score)
                    if score != 1:
                        break

                if body_scores and len(body_scores) == sum(body_scores):
                    measurement_score = add_stage(
                        session=session,
                        history=history,
                        inference=inference,
                        scoring=scoring,
                        scorer_audit=scorer_audit,
                        turn=case.measurement,
                    )

                    if measurement_score == 1:
                        add_stage(
                            session=session,
                            history=history,
                            inference=inference,
                            scoring=scoring,
                            scorer_audit=scorer_audit,
                            turn=case.final,
                        )
                        if task in MEASUREMENT_TASKS:
                            scoring["stage-measured_value"] = "deterministic"

    inference["scorer_audit"] = scorer_audit
    return inference, scoring


def validate_inputs(args: argparse.Namespace) -> CXReasonBenchPath1Loader:
    required = [
        args.benchmark_root / "pnt_on_cxr",
        args.model_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    loader = CXReasonBenchPath1Loader(args.benchmark_root, args.mimic_root)
    loader.validate_cases(args.tasks, expected_per_task=100, require_images=True)
    return loader


def run_metrics(args: argparse.Namespace, inference_root: Path, scoring_root: Path) -> None:
    metric_script = (
        Path(__file__).resolve().parents[1]
        / "external/CXReasonBench/evaluation/metric.py"
    )
    command = [
        sys.executable,
        metric_script.as_posix(),
        "--saved_dir_inference",
        inference_root.as_posix(),
        "--saved_dir_scoring",
        scoring_root.as_posix(),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    metrics_path = args.output_dir / "metrics.txt"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(completed.stdout, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.limit_per_task is not None and args.limit_per_task <= 0:
        raise ValueError("--limit-per-task must be positive")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    set_seed(args.seed)
    loader = validate_inputs(args)

    inference_root = args.output_dir / "inference/reasoning/medgemma-4b-it"
    scoring_root = args.output_dir / "scoring/reasoning/deterministic_mcq_v1/medgemma-4b-it"
    inference_root.mkdir(parents=True, exist_ok=True)
    scoring_root.mkdir(parents=True, exist_ok=True)

    cases = list(
        loader.iter_case_ids(args.tasks, limit_per_task=args.limit_per_task)
    )

    pending = []
    for task, dicom in cases:
        output_path = inference_root / task / f"{dicom}.json"
        score_path = scoring_root / task / f"{dicom}.json"
        if args.resume and output_path.is_file() and score_path.is_file():
            continue
        pending.append((task, dicom))

    config = {
        "model_id": "google/medgemma-4b-it",
        "model_path": str(args.model_path),
        "model_revision": args.model_path.name,
        "img_size": args.img_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "shot": None,
        "evaluation_path": "reasoning",
        "cxreasonbench_base_dir": args.benchmark_root.as_posix(),
        "qa_base_dir": (args.benchmark_root / "qa").as_posix(),
        "mimic_cxr_base": args.mimic_root.as_posix(),
        "scorer": "deterministic_mcq_v1",
        "published_scorer": "gemini-2.0-flash (retired; not used)",
        "selected_tasks": args.tasks,
        "limit_per_task": args.limit_per_task,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    atomic_json(inference_root / "config.json", config)

    print(f"Selected cases: {len(cases)}")
    print(f"Already complete: {len(cases) - len(pending)}")
    print(f"Pending: {len(pending)}")
    print(f"Output: {args.output_dir}")
    print(
        "Initial ETA from the RTX 3080 pilot: "
        f"{human_duration(args.estimate_seconds_per_case * len(pending))}"
    )
    if not pending:
        if args.run_upstream_metrics:
            run_metrics(args, inference_root, scoring_root)
        return

    print("Loading local MedGemma 4B...")
    load_started = time.perf_counter()
    session = MedGemmaSession(args.model_path, args.img_size, args.max_new_tokens)
    load_seconds = time.perf_counter() - load_started
    print(f"Model loaded in {human_duration(load_seconds)}")

    durations: list[float] = []
    run_started = time.perf_counter()
    for index, (task, dicom) in enumerate(pending, start=1):
        case_started = time.perf_counter()
        try:
            inference, scoring = run_case(
                session=session,
                loader=loader,
                task=task,
                dicom=dicom,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA ran out of memory. Close other GPU workloads or rerun with a smaller "
                "--max-new-tokens value; that changes the paper settings and must be reported."
            ) from None

        atomic_json(inference_root / task / f"{dicom}.json", inference)
        atomic_json(scoring_root / task / f"{dicom}.json", scoring)
        duration = time.perf_counter() - case_started
        durations.append(duration)
        remaining = len(pending) - index
        eta = (sum(durations) / len(durations)) * remaining
        print(
            f"[{index}/{len(pending)}] {task}/{dicom} "
            f"{duration:.1f}s | ETA {human_duration(eta)}",
            flush=True,
        )

    elapsed = time.perf_counter() - run_started
    summary = {
        "cases_selected": len(cases),
        "cases_processed_this_run": len(pending),
        "model_load_seconds": load_seconds,
        "inference_seconds": elapsed,
        "mean_case_seconds": sum(durations) / len(durations),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "scorer": "deterministic_mcq_v1",
    }
    atomic_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))

    if args.run_upstream_metrics:
        run_metrics(args, inference_root, scoring_root)

    del session
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

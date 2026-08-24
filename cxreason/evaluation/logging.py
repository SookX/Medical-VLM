from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from cxreason.controller.state import ControllerState
from cxreason.gates.base import GateResult


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def gate_result_to_dict(result: GateResult) -> dict[str, Any]:
    return _json_safe(result)


def iter_attempt_records(
    state: ControllerState,
    *,
    run_id: str,
    mode: str,
    verification_levels: dict[str, str] | None = None,
    corrupted_stage: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> Iterable[dict[str, Any]]:
    for stage, outputs in state.stage_outputs.items():
        for idx, output in enumerate(outputs):
            gate_result = state.gate_results[stage][idx]
            record = {
                "run_id": run_id,
                "mode": mode,
                "case_id": state.case_id,
                "task": state.task,
                "stage": stage,
                "attempt": idx + 1,
                "corrupted_stage": corrupted_stage,
                "verification_level": (verification_levels or {}).get(stage),
                "output": _json_safe(output),
                "gate_result": gate_result_to_dict(gate_result),
                "accepted": state.accepted_outputs.get(stage) == output,
                "case_passed": state.passed,
                "failed_stage": state.failed_stage,
                "model_calls": state.model_calls,
                "verifier_calls": state.verifier_calls,
                "generated_tokens": state.generated_tokens,
            }
            if extra_fields:
                record.update(extra_fields)
            yield record


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def append_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cxreason.evaluation.logging import write_json


def _run(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout + result.stderr).strip()
    return output or None


def collect_run_metadata(config: dict[str, Any] | None = None, cwd: str | Path = ".") -> dict[str, Any]:
    cwd = Path(cwd).resolve()
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(cwd),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config or {},
    }

    metadata["git_head"] = _run(
        ["git", "-c", f"safe.directory={cwd.as_posix()}", "rev-parse", "HEAD"],
        cwd,
    )
    metadata["git_status_short"] = _run(
        ["git", "-c", f"safe.directory={cwd.as_posix()}", "status", "--short"],
        cwd,
    )

    try:
        import torch

        metadata["torch_version"] = torch.__version__
        metadata["cuda_available"] = torch.cuda.is_available()
        metadata["cuda_device_count"] = torch.cuda.device_count()
        metadata["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # pragma: no cover - environment dependent
        metadata["torch_error"] = str(exc)

    try:
        import transformers

        metadata["transformers_version"] = transformers.__version__
    except Exception as exc:  # pragma: no cover - environment dependent
        metadata["transformers_error"] = str(exc)

    return metadata


def write_run_metadata(output_dir: str | Path, config: dict[str, Any] | None = None) -> None:
    output_dir = Path(output_dir)
    write_json(output_dir / "environment.json", collect_run_metadata(config=config))


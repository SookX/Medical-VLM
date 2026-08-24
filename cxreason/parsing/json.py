from __future__ import annotations

import json
import re
from typing import Any


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""

    candidates = [match.group(1) for match in _FENCED_JSON_RE.finditer(text)]
    candidates.append(text)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        stripped = candidate.strip()
        for idx, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(stripped[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    raise ValueError("No JSON object found in model response")


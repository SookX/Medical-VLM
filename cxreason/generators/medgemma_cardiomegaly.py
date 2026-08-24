from __future__ import annotations

from pathlib import Path
from typing import Any

from cxreason.modeling.medgemma import MedGemmaAdapter
from cxreason.parsing.json import extract_json_object
from cxreason.prompting.cardiomegaly import build_cardiomegaly_stage_prompt


class MedGemmaCardiomegalyGenerator:
    """Generate structured cardiomegaly Path-1 stages with MedGemma."""

    def __init__(self, model: MedGemmaAdapter, image_path: str | Path) -> None:
        self.model = model
        self.image_path = Path(image_path)

    def generate(
        self,
        stage: str,
        accepted_outputs: dict[str, Any],
        repair_feedback: str | None = None,
    ) -> dict[str, Any]:
        prompt = build_cardiomegaly_stage_prompt(stage, accepted_outputs, repair_feedback)
        raw_text = self.model.generate(self.image_path, prompt)
        try:
            parsed = extract_json_object(raw_text)
        except ValueError as exc:
            return {"_raw_text": raw_text, "_parse_error": str(exc)}
        parsed["_raw_text"] = raw_text
        return parsed

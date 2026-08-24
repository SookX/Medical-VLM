from __future__ import annotations

from pathlib import Path
from typing import Any

from cxreason.modeling.medgemma import MedGemmaAdapter
from cxreason.parsing.json import extract_json_object
from cxreason.prompting.tasks import build_task_stage_prompt


class MedGemmaTaskGenerator:
    """Generate structured task stages with optional image input."""

    def __init__(
        self,
        task: str,
        model: MedGemmaAdapter,
        image_path: str | Path | None = None,
        development_context: dict[str, Any] | None = None,
    ) -> None:
        self.task = task
        self.model = model
        self.image_path = Path(image_path) if image_path else None
        self.development_context = development_context or {}

    def generate(
        self,
        stage: str,
        accepted_outputs: dict[str, Any],
        repair_feedback: str | None = None,
    ) -> dict[str, Any]:
        prompt = build_task_stage_prompt(
            self.task,
            stage,
            accepted_outputs,
            repair_feedback=repair_feedback,
            development_context=self.development_context.get(stage),
        )
        raw_text = (
            self.model.generate(self.image_path, prompt)
            if self.image_path
            else self.model.generate_text(prompt)
        )
        try:
            parsed = extract_json_object(raw_text)
        except ValueError as exc:
            return {"_raw_text": raw_text, "_parse_error": str(exc)}
        parsed["_raw_text"] = raw_text
        return parsed

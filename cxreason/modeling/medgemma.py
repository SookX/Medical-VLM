"""Minimal MedGemma image-text adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import pipeline


@dataclass(frozen=True)
class MedGemmaConfig:
    model_id: str = "google/medgemma-4b-it"
    task: str = "image-text-to-text"
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    max_new_tokens: int = 512
    temperature: float = 0.0
    system_prompt: str = (
        "You are a medical imaging reasoning assistant for research evaluation. "
        "Follow the requested CXReasonBench stage format exactly."
    )


class MedGemmaAdapter:
    """Thin wrapper around the Transformers image-text pipeline."""

    def __init__(self, config: MedGemmaConfig | None = None) -> None:
        self.config = config or MedGemmaConfig()
        torch_dtype = getattr(torch, self.config.torch_dtype)
        self.pipe = pipeline(
            self.config.task,
            model=self.config.model_id,
            dtype=torch_dtype,
            device=self.config.device,
        )

    def generate(self, image_path: str | Path, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        output = self.pipe(
            text=messages,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return output[0]["generated_text"][-1]["content"]

    def generate_text(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.config.system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]
        output = self.pipe(
            text=messages,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return output[0]["generated_text"][-1]["content"]

"""Native multimodal MedGemma session used by the CXReasonBench runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


SYSTEM_MESSAGE = (
    "You are a well-trained radiologist answering multiple-choice questions about a chest X-ray. "
    "Base your assessment solely on the chest X-ray itself, without considering "
    "labels or annotations in the upper corners. "
    "Rely purely on anatomical and radiographic features. "
    "Provide your response in the format: FINAL ANSWER: [YOUR FINAL ANSWER], "
    "including the letter of your choice followed by the selected option "
    "(e.g., FINAL ANSWER: (A) Yes). "
    "Always state your selected option first, then explain your reasoning if needed."
)

HistoryEntry = tuple[str, list[Path], str]


@dataclass(frozen=True)
class MedGemmaGeneration:
    text: str
    input_tokens: int
    generated_tokens: int


def build_native_conversation(
    *,
    query: str,
    image_paths: list[Path],
    history: list[HistoryEntry],
    system_message: str = SYSTEM_MESSAGE,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Build the exact native chat structure and ordered image list."""

    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": system_message}
    ]
    all_image_paths: list[Path] = []
    for old_query, old_images, old_response in history:
        all_image_paths.extend(old_images)
        conversation.extend(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": old_query}]
                    + [{"type": "image"} for _ in old_images],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": old_response}],
                },
            ]
        )
    all_image_paths.extend(image_paths)
    conversation.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": query}]
            + [{"type": "image"} for _ in image_paths],
        }
    )
    return conversation, all_image_paths


def resize_image(path: Path, new_size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if max(width, height) <= new_size:
        return image
    if width >= height:
        target = (new_size, int(new_size * height / width))
    else:
        target = (int(new_size * width / height), new_size)
    return image.resize(target)


class MedGemmaSession:
    """Persistent model/processor pair preserving full multimodal history."""

    def __init__(self, model_path: Path, img_size: int, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.img_size = img_size
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def generate(
        self,
        *,
        query: str,
        image_paths: list[Path],
        history: list[HistoryEntry],
    ) -> str:
        return self.generate_with_metadata(
            query=query, image_paths=image_paths, history=history
        ).text

    def generate_with_metadata(
        self,
        *,
        query: str,
        image_paths: list[Path],
        history: list[HistoryEntry],
        temperature: float = 0.0,
        top_p: float = 1.0,
        sampling_seed: int | None = None,
    ) -> MedGemmaGeneration:
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be between 0 and 1")
        torch = self._torch
        conversation, all_image_paths = build_native_conversation(
            query=query, image_paths=image_paths, history=history
        )
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        images = [resize_image(path, self.img_size) for path in all_image_paths]
        if images:
            inputs = self.processor(
                images=images, text=prompt, return_tensors="pt", padding=True
            )
        else:
            inputs = self.processor(text=prompt, return_tensors="pt", padding=True)

        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if torch.is_floating_point(value):
                inputs[key] = value.to(self.device, dtype=torch.bfloat16)
            else:
                inputs[key] = value.to(self.device)

        input_tokens = inputs["input_ids"].shape[1]
        do_sample = temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs.update(
                {"temperature": temperature, "top_p": top_p}
            )
            if sampling_seed is None:
                raise ValueError("sampling_seed is required when sampling")
            torch.manual_seed(sampling_seed)
            torch.cuda.manual_seed_all(sampling_seed)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        response = self.processor.decode(
            output[0][input_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        generated_tokens = int(output.shape[1] - input_tokens)
        for image in images:
            image.close()
        del inputs, output, images
        return MedGemmaGeneration(
            text=response,
            input_tokens=int(input_tokens),
            generated_tokens=generated_tokens,
        )

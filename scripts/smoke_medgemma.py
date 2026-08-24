from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import pipeline


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a MedGemma 4B smoke test.")
    parser.add_argument("--config", default="configs/medgemma.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--prompt",
        default="Describe the chest X-ray findings relevant to cardiomegaly in one concise paragraph.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    model_config = config["model"]

    dtype_name = model_config.get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, dtype_name)

    image = Image.open(args.image).convert("RGB")
    pipe = pipeline(
        model_config.get("task", "image-text-to-text"),
        model=model_config["model_id"],
        torch_dtype=torch_dtype,
        device=model_config.get("device", "cuda"),
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": config["prompting"]["system_prompt"]}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]

    output = pipe(
        text=messages,
        max_new_tokens=model_config.get("max_new_tokens", 512),
        temperature=model_config.get("temperature", 0.0),
    )
    print(output[0]["generated_text"][-1]["content"])


if __name__ == "__main__":
    main()

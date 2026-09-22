"""Versioned adapter for the benchmark-linked CXAS anatomy model.

CXAS is used only to reproduce the *distribution* of displayed Stage-2
candidate masks on an external cohort.  It is never used as the independent
runtime verifier; that role remains with ChestX-Det.
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from cxreason.vision.chestx_det import sha256_file


@dataclass(frozen=True)
class CXASPrediction:
    """Selected binary CXAS masks in the model's square center-crop space."""

    original_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    masks: dict[str, np.ndarray]
    logit_threshold: float
    model_version: str
    weights_sha256: str
    source_sha256: str

    def __post_init__(self) -> None:
        width, height = self.original_size
        left, top, right, bottom = self.crop_box
        if width <= 0 or height <= 0:
            raise ValueError("original_size must be positive")
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("crop_box must lie inside the original image")
        if not self.masks:
            raise ValueError("At least one CXAS target is required")
        shapes = {np.asarray(mask).shape for mask in self.masks.values()}
        if len(shapes) != 1 or any(len(shape) != 2 for shape in shapes):
            raise ValueError("All CXAS masks must share one 2-D shape")
        if len(self.weights_sha256) != 64 or len(self.source_sha256) != 64:
            raise ValueError("Prediction hashes must be SHA-256 hex digests")

    def combined_mask(self, targets: str | Iterable[str]) -> np.ndarray:
        names = (targets,) if isinstance(targets, str) else tuple(targets)
        if not names:
            raise ValueError("At least one target is required")
        missing = [name for name in names if name not in self.masks]
        if missing:
            raise KeyError(f"Prediction does not contain targets: {missing}")
        return np.logical_or.reduce(
            [np.asarray(self.masks[name], dtype=bool) for name in names]
        )

    def project_mask(
        self, targets: str | Iterable[str], output_size: tuple[int, int]
    ) -> np.ndarray:
        """Project one or more masks into an uncropped output-image frame."""

        combined = self.combined_mask(targets)
        output_width, output_height = output_size
        if output_width <= 0 or output_height <= 0:
            raise ValueError("output_size must be positive")
        original_width, original_height = self.original_size
        left, top, right, bottom = self.crop_box
        projected_left = round(left * output_width / original_width)
        projected_right = round(right * output_width / original_width)
        projected_top = round(top * output_height / original_height)
        projected_bottom = round(bottom * output_height / original_height)
        projected_left = min(max(projected_left, 0), output_width - 1)
        projected_top = min(max(projected_top, 0), output_height - 1)
        projected_right = min(max(projected_right, projected_left + 1), output_width)
        projected_bottom = min(max(projected_bottom, projected_top + 1), output_height)
        resized = Image.fromarray(combined.astype(np.uint8) * 255).resize(
            (projected_right - projected_left, projected_bottom - projected_top),
            resample=Image.Resampling.NEAREST,
        )
        output = np.zeros((output_height, output_width), dtype=bool)
        output[
            projected_top:projected_bottom, projected_left:projected_right
        ] = np.asarray(resized) > 0
        return output


class CXASSegmenter:
    """Run TorchXRayVision CXAS with explicit target and provenance checks."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        device: str = "cuda",
        logit_threshold: float = 0.0,
        targets: Iterable[str],
    ) -> None:
        import torch
        import torchxrayvision as xrv

        selected = tuple(dict.fromkeys(targets))
        if not selected:
            raise ValueError("At least one CXAS target is required")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(device)
        self.logit_threshold = float(logit_threshold)
        self.model_version = importlib.metadata.version("torchxrayvision")
        self.model = xrv.baseline_models.chestx_anatomy.UNetResNet50(
            cache_dir=str(cache_dir)
        ).to(self.device)
        self.model.eval()
        unknown = set(selected) - set(self.model.targets)
        if unknown:
            raise ValueError(f"Invalid CXAS targets: {sorted(unknown)}")
        self.targets = selected
        self.target_indices = {
            name: self.model.targets.index(name) for name in self.targets
        }
        self.weights_path = Path(self.model.weights_filename_local).resolve()
        self.weights_sha256 = sha256_file(self.weights_path)

    @staticmethod
    def _center_crop(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError("Expected normalized image shaped [1, height, width]")
        _, height, width = image.shape
        size = min(height, width)
        left = width // 2 - size // 2
        top = height // 2 - size // 2
        return image[:, top : top + size, left : left + size], (
            left,
            top,
            left + size,
            top + size,
        )

    def predict(self, image_path: Path) -> CXASPrediction:
        import torch
        import torchxrayvision as xrv

        image = xrv.utils.load_image(str(image_path)).astype(np.float32, copy=False)
        _, height, width = image.shape
        cropped, crop_box = self._center_crop(image)
        tensor = torch.from_numpy(cropped).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)[0].detach().cpu()
        masks = {
            name: (logits[index].numpy() >= self.logit_threshold)
            for name, index in self.target_indices.items()
        }
        return CXASPrediction(
            original_size=(width, height),
            crop_box=crop_box,
            masks=masks,
            logit_threshold=self.logit_threshold,
            model_version=self.model_version,
            weights_sha256=self.weights_sha256,
            source_sha256=sha256_file(image_path),
        )


def save_cxas_prediction(path: Path, prediction: CXASPrediction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "original_size": list(prediction.original_size),
        "crop_box": list(prediction.crop_box),
        "targets": list(prediction.masks),
        "logit_threshold": prediction.logit_threshold,
        "model_version": prediction.model_version,
        "weights_sha256": prediction.weights_sha256,
        "source_sha256": prediction.source_sha256,
    }
    arrays = {
        f"mask_{index}": np.asarray(prediction.masks[name], dtype=np.uint8)
        for index, name in enumerate(prediction.masks)
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            **arrays,
        )
    temporary.replace(path)


def load_cxas_prediction(path: Path) -> CXASPrediction:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("schema_version") != 1:
            raise RuntimeError(f"Unsupported CXAS prediction schema in {path}")
        masks = {
            name: np.asarray(archive[f"mask_{index}"], dtype=bool)
            for index, name in enumerate(metadata["targets"])
        }
    return CXASPrediction(
        original_size=tuple(int(value) for value in metadata["original_size"]),
        crop_box=tuple(int(value) for value in metadata["crop_box"]),
        masks=masks,
        logit_threshold=float(metadata["logit_threshold"]),
        model_version=str(metadata["model_version"]),
        weights_sha256=str(metadata["weights_sha256"]),
        source_sha256=str(metadata["source_sha256"]),
    )

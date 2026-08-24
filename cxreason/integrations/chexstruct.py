"""Raw CSV access for the CheXStruct Hugging Face dataset.

The dataset repository contains one CSV per source dataset and task. The task
files intentionally have different schemas, so callers should load individual
CSV files instead of using one flat Hugging Face `datasets` table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SOURCE_DATASETS = ("nih_cxr14", "openi", "vindrcxr")

TASK_FILES = {
    "abdominal_xray": "abdomial_xray.csv",
    "aortic_knob_enlargement": "aortic_knob_enlargement.csv",
    "ascending_aorta_enlargement": "ascending_aorta_enlargement.csv",
    "cardiomegaly": "cardiomegaly.csv",
    "carina_angle": "carina_angle.csv",
    "descending_aorta_enlargement": "descending_aorta_enlargement.csv",
    "descending_aorta_tortuous": "descending_aorta_tortuous.csv",
    "inclusion": "inclusion.csv",
    "inspiration": "inspiration.csv",
    "mask_number": "mask_number.csv",
    "mediastinal_widening": "mediastinal_widening.csv",
    "projection": "projection.csv",
    "rotation": "rotation.csv",
    "trachea_deviation": "trachea_deviation.csv",
    "window": "window.csv",
}


@dataclass(frozen=True)
class CheXStructTable:
    source_dataset: str
    task: str
    path: Path


class CheXStructRepository:
    """Discover and load CheXStruct task CSV files from a local clone."""

    def __init__(self, root: str | Path = "data/CheXStruct") -> None:
        self.root = Path(root)

    def table_path(self, source_dataset: str, task: str) -> Path:
        if source_dataset not in SOURCE_DATASETS:
            raise ValueError(f"Unknown CheXStruct source dataset: {source_dataset}")
        try:
            filename = TASK_FILES[task]
        except KeyError as exc:
            raise ValueError(f"Unknown CheXStruct task: {task}") from exc
        return self.root / source_dataset / filename

    def iter_tables(self) -> Iterable[CheXStructTable]:
        for source_dataset in SOURCE_DATASETS:
            for task in TASK_FILES:
                path = self.table_path(source_dataset, task)
                if path.exists():
                    yield CheXStructTable(source_dataset, task, path)

    def load_table(self, source_dataset: str, task: str) -> pd.DataFrame:
        path = self.table_path(source_dataset, task)
        if not path.exists():
            raise FileNotFoundError(f"CheXStruct CSV not found: {path}")
        return pd.read_csv(path)


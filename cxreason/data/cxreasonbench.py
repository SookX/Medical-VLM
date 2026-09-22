"""Typed loader for the native CXReasonBench Path-1 questions and images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


PATH1_TASKS = (
    "aortic_knob_enlargement",
    "ascending_aorta_enlargement",
    "cardiomegaly",
    "carina_angle",
    "descending_aorta_enlargement",
    "descending_aorta_tortuous",
    "inclusion",
    "inspiration",
    "mediastinal_widening",
    "projection",
    "rotation",
    "trachea_deviation",
)

MEASUREMENT_TASKS = frozenset(
    {
        "rotation",
        "projection",
        "cardiomegaly",
        "mediastinal_widening",
        "carina_angle",
        "aortic_knob_enlargement",
        "descending_aorta_enlargement",
        "descending_aorta_tortuous",
    }
)


class CXReasonBenchDataError(RuntimeError):
    """Raised when the native benchmark data violates the expected schema."""


@dataclass(frozen=True)
class NativeStageTurn:
    """One unmodified CXReasonBench question in a Path-1 conversation."""

    key: str
    scorer_stage: str
    question: str
    answer: str
    image_paths: tuple[Path, ...]
    source_path: Path


@dataclass(frozen=True)
class NativePath1Case:
    """All possible native Path-1 turns for one task/DICOM pair."""

    task: str
    dicom: str
    cxr_path: str
    initial: NativeStageTurn
    criteria: tuple[NativeStageTurn, ...]
    criterion_refinement: NativeStageTurn | None
    anatomy: tuple[NativeStageTurn, ...]
    measurement: NativeStageTurn
    final: NativeStageTurn

    @property
    def all_turns(self) -> tuple[NativeStageTurn, ...]:
        turns = [self.initial, *self.criteria]
        if self.criterion_refinement is not None:
            turns.append(self.criterion_refinement)
        turns.extend(self.anatomy)
        turns.extend((self.measurement, self.final))
        return tuple(turns)

    @property
    def qa_files(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(turn.source_path for turn in self.all_turns))

    @property
    def image_paths(self) -> tuple[Path, ...]:
        return tuple(
            dict.fromkeys(
                image for turn in self.all_turns for image in turn.image_paths
            )
        )

    def turn(self, key: str) -> NativeStageTurn:
        for turn in self.all_turns:
            if turn.key == key:
                return turn
        raise KeyError(key)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CXReasonBenchDataError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CXReasonBenchDataError(f"Expected a JSON object in {path}")
    return value


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _question_pairs(payload: dict[str, Any], source: Path) -> list[tuple[str, str]]:
    questions = _as_list(payload.get("question"))
    answers = _as_list(payload.get("answer"))
    if not questions or len(questions) != len(answers):
        raise CXReasonBenchDataError(
            f"Question/answer count mismatch in {source}: "
            f"{len(questions)} questions and {len(answers)} answers"
        )
    pairs: list[tuple[str, str]] = []
    for question, answer in zip(questions, answers):
        if not isinstance(question, str) or not isinstance(answer, str):
            raise CXReasonBenchDataError(f"Non-string question or answer in {source}")
        pairs.append((question, answer))
    return pairs


class CXReasonBenchPath1Loader:
    """Load native Path-1 cases without selecting a trajectory or scoring it."""

    def __init__(self, benchmark_root: Path, mimic_root: Path) -> None:
        self.benchmark_root = benchmark_root.resolve()
        self.mimic_root = mimic_root.resolve()
        self.qa_root = self.benchmark_root / "qa"
        self.segmask_root = self.benchmark_root / "segmask_bodypart"
        self.index_path = self.benchmark_root / "dx_by_dicoms.json"
        required = (self.qa_root, self.segmask_root, self.index_path, self.mimic_root)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise CXReasonBenchDataError(
                "Missing native Path-1 inputs:\n" + "\n".join(missing)
            )
        self._index = _read_json(self.index_path)
        self._cache: dict[tuple[str, str], NativePath1Case] = {}

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(task for task in PATH1_TASKS if task in self._index)

    def case_ids(self, task: str) -> tuple[str, ...]:
        if task not in PATH1_TASKS:
            raise CXReasonBenchDataError(f"Unsupported Path-1 task: {task}")
        values = self._index.get(task)
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise CXReasonBenchDataError(f"Invalid DICOM index for task {task}")
        return tuple(values)

    def iter_case_ids(
        self, tasks: Iterable[str] = PATH1_TASKS, *, limit_per_task: int | None = None
    ) -> Iterator[tuple[str, str]]:
        if limit_per_task is not None and limit_per_task <= 0:
            raise ValueError("limit_per_task must be positive")
        for task in tasks:
            dicoms = self.case_ids(task)
            if limit_per_task is not None:
                dicoms = dicoms[:limit_per_task]
            yield from ((task, dicom) for dicom in dicoms)

    def _stage_file(self, task: str, stage: str, dicom: str) -> Path:
        matches = sorted(
            (self.qa_root / task / "path1" / stage).glob(f"*/{dicom}.json")
        )
        if len(matches) != 1:
            raise CXReasonBenchDataError(
                f"Expected one QA file for {task}/{stage}/{dicom}, "
                f"found {len(matches)}"
            )
        return matches[0]

    def _single_text_turn(
        self, *, key: str, scorer_stage: str, source: Path
    ) -> NativeStageTurn:
        pairs = _question_pairs(_read_json(source), source)
        if len(pairs) != 1:
            raise CXReasonBenchDataError(f"Expected one question in {source}")
        question, answer = pairs[0]
        return NativeStageTurn(key, scorer_stage, question, answer, (), source)

    def load_case(self, task: str, dicom: str) -> NativePath1Case:
        cache_key = (task, dicom)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if dicom not in self.case_ids(task):
            raise CXReasonBenchDataError(f"Unknown case for {task}: {dicom}")

        init_path = self.qa_root / task / "path1/init/basic" / f"{dicom}.json"
        init_payload = _read_json(init_path)
        init_pairs = _question_pairs(init_payload, init_path)
        if len(init_pairs) != 1:
            raise CXReasonBenchDataError(f"Expected one initial question in {init_path}")
        relative_images = init_payload.get("img_path")
        if not isinstance(relative_images, list) or not relative_images or not all(
            isinstance(value, str) for value in relative_images
        ):
            raise CXReasonBenchDataError(f"Invalid initial img_path in {init_path}")
        initial = NativeStageTurn(
            "init",
            "init",
            init_pairs[0][0],
            init_pairs[0][1],
            tuple(self.mimic_root / value for value in relative_images),
            init_path,
        )

        criteria_path = self._stage_file(task, "stage1", dicom)
        criteria = tuple(
            NativeStageTurn(
                f"criteria_{index}", "criteria", question, answer, (), criteria_path
            )
            for index, (question, answer) in enumerate(
                _question_pairs(_read_json(criteria_path), criteria_path)
            )
        )

        refinement_path = (
            self.qa_root / task / "path1/stage1.5/basic" / f"{dicom}.json"
        )
        criterion_refinement = (
            self._single_text_turn(
                key="custom_criteria",
                scorer_stage="custom_criteria",
                source=refinement_path,
            )
            if refinement_path.is_file()
            else None
        )

        anatomy_path = self._stage_file(task, "stage2", dicom)
        anatomy_payload = _read_json(anatomy_path)
        anatomy_pairs = _question_pairs(anatomy_payload, anatomy_path)
        image_groups = anatomy_payload.get("img_path")
        if not isinstance(image_groups, list) or len(image_groups) != len(anatomy_pairs):
            raise CXReasonBenchDataError(
                f"Stage-2 image-group mismatch in {anatomy_path}"
            )
        anatomy_turns: list[NativeStageTurn] = []
        for index, ((question, answer), image_group) in enumerate(
            zip(anatomy_pairs, image_groups)
        ):
            if not isinstance(image_group, list) or not all(
                isinstance(value, str) for value in image_group
            ):
                raise CXReasonBenchDataError(
                    f"Invalid Stage-2 image group in {anatomy_path}"
                )
            anatomy_turns.append(
                NativeStageTurn(
                    f"bodypart_{index}",
                    "bodypart",
                    question,
                    answer,
                    tuple(
                        self.segmask_root / value.lstrip("/\\")
                        for value in image_group
                    ),
                    anatomy_path,
                )
            )

        measurement = self._single_text_turn(
            key="measurement",
            scorer_stage="measurement",
            source=self.qa_root
            / task
            / "path1/stage3/basic"
            / f"{dicom}.json",
        )
        final = self._single_text_turn(
            key="final",
            scorer_stage="final",
            source=self.qa_root
            / task
            / "path1/stage4/basic"
            / f"{dicom}.json",
        )
        case = NativePath1Case(
            task=task,
            dicom=dicom,
            cxr_path=relative_images[0],
            initial=initial,
            criteria=criteria,
            criterion_refinement=criterion_refinement,
            anatomy=tuple(anatomy_turns),
            measurement=measurement,
            final=final,
        )
        self._cache[cache_key] = case
        return case

    def validate_cases(
        self,
        tasks: Iterable[str] = PATH1_TASKS,
        *,
        expected_per_task: int | None = None,
        require_images: bool = True,
    ) -> dict[str, int]:
        """Load all requested cases and optionally verify every image exists."""

        counts: dict[str, int] = {}
        for task in tasks:
            dicoms = self.case_ids(task)
            if expected_per_task is not None and len(dicoms) != expected_per_task:
                raise CXReasonBenchDataError(
                    f"Expected {expected_per_task} cases for {task}, found {len(dicoms)}"
                )
            counts[task] = len(dicoms)
            for dicom in dicoms:
                case = self.load_case(task, dicom)
                if require_images:
                    missing = [str(path) for path in case.image_paths if not path.is_file()]
                    if missing:
                        raise CXReasonBenchDataError(
                            f"Missing images for {task}/{dicom}:\n" + "\n".join(missing)
                        )
        return counts


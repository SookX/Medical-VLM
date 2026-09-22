from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cxreason.data.cxreasonbench import (
    CXReasonBenchDataError,
    CXReasonBenchPath1Loader,
    PATH1_TASKS,
)
from cxreason.modeling.medgemma_session import (
    SYSTEM_MESSAGE,
    build_native_conversation,
    resize_image,
)
from scripts.run_medgemma4b_path1_reproduction import run_case


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class NativePath1LoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.benchmark = self.root / "benchmark"
        self.mimic = self.root / "mimic"
        self.task = "cardiomegaly"
        self.dicom = "case-001"
        self._make_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _make_fixture(self) -> None:
        qa = self.benchmark / "qa" / self.task / "path1"
        original = self.mimic / "p10/original.jpg"
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(b"original")
        for name in ("one.jpg", "two.jpg", "three.jpg"):
            image = self.benchmark / "segmask_bodypart/masks" / name
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(name.encode("ascii"))
        _write_json(self.benchmark / "dx_by_dicoms.json", {self.task: [self.dicom]})
        _write_json(
            qa / "init/basic" / f"{self.dicom}.json",
            {
                "question": "Initial question",
                "answer": "(a) Yes",
                "img_path": ["p10/original.jpg"],
            },
        )
        _write_json(
            qa / "stage1/two-round" / f"{self.dicom}.json",
            {
                "question": ["Criterion round one", "Criterion round two"],
                "answer": ["(e) Need new options", "(b) CTR"],
            },
        )
        _write_json(
            qa / "stage1.5/basic" / f"{self.dicom}.json",
            {"question": "Adopt criterion?", "answer": "(a) Yes"},
        )
        _write_json(
            qa / "stage2/two-round_partial_inclusion" / f"{self.dicom}.json",
            {
                "question": ["Anatomy round one", "Anatomy round two"],
                "answer": ["(e) Need new options", "(a) First"],
                "img_path": [
                    ["masks/one.jpg", "masks/two.jpg"],
                    ["masks/three.jpg"],
                ],
            },
        )
        _write_json(
            qa / "stage3/basic" / f"{self.dicom}.json",
            {"question": "Measurement question", "answer": "(b) Normal"},
        )
        _write_json(
            qa / "stage4/basic" / f"{self.dicom}.json",
            {"question": "Final question", "answer": "(a) No"},
        )

    def test_loads_all_native_rounds_in_order(self) -> None:
        loader = CXReasonBenchPath1Loader(self.benchmark, self.mimic)
        case = loader.load_case(self.task, self.dicom)
        self.assertEqual(
            [turn.key for turn in case.all_turns],
            [
                "init",
                "criteria_0",
                "criteria_1",
                "custom_criteria",
                "bodypart_0",
                "bodypart_1",
                "measurement",
                "final",
            ],
        )
        self.assertEqual(case.criteria[1].question, "Criterion round two")
        self.assertEqual(case.criterion_refinement.question, "Adopt criterion?")
        self.assertEqual(
            [path.name for path in case.anatomy[0].image_paths],
            ["one.jpg", "two.jpg"],
        )
        self.assertIs(loader.load_case(self.task, self.dicom), case)

    def test_validates_every_referenced_image(self) -> None:
        loader = CXReasonBenchPath1Loader(self.benchmark, self.mimic)
        self.assertEqual(
            loader.validate_cases([self.task], expected_per_task=1),
            {self.task: 1},
        )
        (self.benchmark / "segmask_bodypart/masks/three.jpg").unlink()
        with self.assertRaisesRegex(CXReasonBenchDataError, "Missing images"):
            loader.validate_cases([self.task], expected_per_task=1)

    def test_rejects_ambiguous_stage_variant(self) -> None:
        source = (
            self.benchmark
            / "qa"
            / self.task
            / "path1/stage1/basic"
            / f"{self.dicom}.json"
        )
        _write_json(source, {"question": "duplicate", "answer": "(a) duplicate"})
        loader = CXReasonBenchPath1Loader(self.benchmark, self.mimic)
        with self.assertRaisesRegex(CXReasonBenchDataError, "found 2"):
            loader.load_case(self.task, self.dicom)

    def test_iter_case_ids_validates_limit(self) -> None:
        loader = CXReasonBenchPath1Loader(self.benchmark, self.mimic)
        self.assertEqual(list(loader.iter_case_ids([self.task])), [(self.task, self.dicom)])
        with self.assertRaises(ValueError):
            list(loader.iter_case_ids([self.task], limit_per_task=0))


class NativeMedGemmaConversationTest(unittest.TestCase):
    def test_preserves_history_and_image_order(self) -> None:
        first, second, current = Path("first.png"), Path("second.png"), Path("current.png")
        history = [
            ("old multimodal", [first, second], "old response"),
            ("old text", [], "second response"),
        ]
        conversation, images = build_native_conversation(
            query="current question", image_paths=[current], history=history
        )
        self.assertEqual(images, [first, second, current])
        self.assertEqual(conversation[0], {"role": "system", "content": SYSTEM_MESSAGE})
        self.assertEqual(
            conversation[1]["content"],
            [
                {"type": "text", "text": "old multimodal"},
                {"type": "image"},
                {"type": "image"},
            ],
        )
        self.assertEqual(conversation[-1]["content"][-1], {"type": "image"})
        self.assertEqual(
            [message["role"] for message in conversation],
            ["system", "user", "assistant", "user", "assistant", "user"],
        )

    def test_resize_preserves_aspect_ratio_and_avoids_upscale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "image.png"
            Image.new("RGB", (200, 100)).save(source)
            resized = resize_image(source, 100)
            self.assertEqual(resized.size, (100, 50))
            resized.close()
            unchanged = resize_image(source, 300)
            self.assertEqual(unchanged.size, (200, 100))
            unchanged.close()


class NativePath1DatasetIntegrationTest(unittest.TestCase):
    workspace = Path(__file__).resolve().parents[1]
    benchmark = workspace / "physionet.org/files/chexstruct-cxreasonbench/1.0.1/CXReasonBench"
    mimic = workspace / "physionet.org/files/mimic-cxr-jpg/2.1.0/files"
    baseline_inference = (
        workspace
        / "outputs/medgemma4b_path1_reproduction/inference/reasoning/medgemma-4b-it"
    )
    baseline_scoring = (
        workspace
        / "outputs/medgemma4b_path1_reproduction/scoring/reasoning/deterministic_mcq_v1/medgemma-4b-it"
    )

    @unittest.skipUnless(
        benchmark.is_dir() and mimic.is_dir(),
        "Downloaded CXReasonBench and MIMIC subset are not available",
    )
    def test_loads_all_1200_cases_and_assets(self) -> None:
        loader = CXReasonBenchPath1Loader(self.benchmark, self.mimic)
        counts = loader.validate_cases(PATH1_TASKS, expected_per_task=100)
        self.assertEqual(sum(counts.values()), 1200)
        cases = [
            loader.load_case(task, dicom)
            for task, dicom in loader.iter_case_ids(PATH1_TASKS)
        ]
        self.assertTrue(any(len(case.criteria) > 1 for case in cases))
        self.assertTrue(any(len(case.anatomy) > 1 for case in cases))
        self.assertTrue(any(case.criterion_refinement is not None for case in cases))

    @unittest.skipUnless(
        benchmark.is_dir()
        and mimic.is_dir()
        and baseline_inference.is_dir()
        and baseline_scoring.is_dir(),
        "Downloaded benchmark and saved baseline are not available",
    )
    def test_refactored_runner_exactly_replays_saved_responses(self) -> None:
        class ReplaySession:
            def __init__(self, inference: dict) -> None:
                self.records = [
                    value
                    for key, value in inference.items()
                    if key.startswith("stage-")
                ]
                self.index = 0

            def generate(self, *, query, image_paths, history) -> str:
                record = self.records[self.index]
                if query != record["query"]:
                    raise AssertionError(
                        f"Question mismatch at replay turn {self.index}"
                    )
                if [str(path) for path in image_paths] != record["img_path"]:
                    raise AssertionError(
                        f"Image mismatch at replay turn {self.index}"
                    )
                if len(history) != self.index:
                    raise AssertionError(
                        f"History length mismatch at replay turn {self.index}"
                    )
                self.index += 1
                return record["response"]

        loader = CXReasonBenchPath1Loader(self.benchmark, self.mimic)
        for task, dicom in loader.iter_case_ids(PATH1_TASKS):
            with self.subTest(task=task, dicom=dicom):
                inference_path = self.baseline_inference / task / f"{dicom}.json"
                scoring_path = self.baseline_scoring / task / f"{dicom}.json"
                expected_inference = json.loads(
                    inference_path.read_text(encoding="utf-8")
                )
                expected_scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
                session = ReplaySession(expected_inference)
                actual_inference, actual_scoring = run_case(
                    session=session,
                    loader=loader,
                    task=task,
                    dicom=dicom,
                )
                normalized_inference = json.loads(json.dumps(actual_inference))
                self.assertEqual(normalized_inference, expected_inference)
                self.assertEqual(actual_scoring, expected_scoring)
                self.assertEqual(session.index, len(session.records))


if __name__ == "__main__":
    unittest.main()

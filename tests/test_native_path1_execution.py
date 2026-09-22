from __future__ import annotations

import json
import unittest
from pathlib import Path

from cxreason.data.cxreasonbench import (
    CXReasonBenchPath1Loader,
    PATH1_TASKS,
    NativePath1Case,
    NativeStageTurn,
)
from cxreason.gates.base import GateResult
from cxreason.path1.execution import NativeGeneration, NativePath1Executor
from cxreason.path1.graph import NativePath1Graph, normalize_repair_stages
from cxreason.path1.history import (
    RepairedStageHistoryFormatter,
    normalized_selected_response,
)
from cxreason.path1.verifiers import OracleAnswerVerifier


def _turn(key: str, stage: str) -> NativeStageTurn:
    return NativeStageTurn(
        key=key,
        scorer_stage=stage,
        question=f"{key}? Options: (a) Correct, (b) Wrong"
        + (", (c) I don't know" if key == "init" else ""),
        answer="(a) Correct",
        image_paths=(Path(f"{key}.png"),) if key in {"init", "bodypart_0"} else (),
        source_path=Path(f"qa/{key}.json"),
    )


def _case() -> NativePath1Case:
    return NativePath1Case(
        task="cardiomegaly",
        dicom="case-001",
        cxr_path="init.png",
        initial=_turn("init", "init"),
        criteria=(_turn("criteria_0", "criteria"),),
        criterion_refinement=_turn("custom_criteria", "custom_criteria"),
        anatomy=(_turn("bodypart_0", "bodypart"),),
        measurement=_turn("measurement", "measurement"),
        final=_turn("final", "final"),
    )


class ScriptedGenerator:
    def __init__(self, responses: dict[str, list[str]] | None = None) -> None:
        self.responses = responses or {}
        self.indices: dict[str, int] = {}
        self.calls: list[dict] = []

    def generate(self, turn, accepted_history, repair_feedback=None):
        index = self.indices.get(turn.key, 0)
        choices = self.responses.get(turn.key, ["FINAL ANSWER: (a) Correct"])
        response = choices[min(index, len(choices) - 1)]
        self.indices[turn.key] = index + 1
        self.calls.append(
            {
                "key": turn.key,
                "history": [entry[2] for entry in accepted_history],
                "feedback": repair_feedback,
            }
        )
        return NativeGeneration(response, generated_tokens=4)


class AlwaysPassVerifier:
    name = "always_pass"

    def verify(self, turn, response, accepted_responses):
        del turn, response, accepted_responses
        return GateResult(True, reason=None, metadata={"verification_level": "test"})


class NativePath1GraphTest(unittest.TestCase):
    def test_builds_dynamic_links_and_optional_stage(self) -> None:
        graph = NativePath1Graph.from_case(_case())
        self.assertEqual(
            [node.turn.key for node in graph.nodes],
            [
                "init",
                "criteria_0",
                "custom_criteria",
                "bodypart_0",
                "measurement",
                "final",
            ],
        )
        self.assertEqual(graph.node("custom_criteria").predecessor, "criteria_0")
        self.assertEqual(graph.node("custom_criteria").successor, "bodypart_0")
        self.assertFalse(graph.node("init").repairable)
        self.assertTrue(graph.node("final").repairable)

    def test_normalizes_paper_stage_aliases(self) -> None:
        self.assertEqual(
            normalize_repair_stages(["stage1", "stage1.5", "stage2", "stage3", "stage4"]),
            frozenset(
                {"criteria", "custom_criteria", "bodypart", "measurement", "final"}
            ),
        )
        self.assertEqual(normalize_repair_stages(["all"]), normalize_repair_stages([
            "criteria", "custom_criteria", "bodypart", "measurement", "final"
        ]))
        with self.assertRaisesRegex(ValueError, "initial diagnostic decision"):
            normalize_repair_stages(["init"])


class NativePath1ExecutorTest(unittest.TestCase):
    def test_no_repair_stops_at_first_failed_native_stage(self) -> None:
        generator = ScriptedGenerator(
            {"criteria_0": ["FINAL ANSWER: (b) Wrong", "FINAL ANSWER: (a) Correct"]}
        )
        state = NativePath1Executor().run(_case(), generator)
        self.assertFalse(state.passed)
        self.assertEqual(state.failed_stage_key, "criteria_0")
        self.assertEqual(state.stop_reason, "native_stage_failed")
        self.assertEqual(state.model_calls, 2)
        self.assertEqual(state.verifier_calls, 0)

    def test_each_stage_can_be_ablated_independently(self) -> None:
        stage_cases = {
            "stage1": ("criteria", "criteria_0"),
            "stage1.5": ("custom_criteria", "custom_criteria"),
            "stage2": ("bodypart", "bodypart_0"),
            "stage3": ("measurement", "measurement"),
            "stage4": ("final", "final"),
        }
        for alias, (group, key) in stage_cases.items():
            with self.subTest(stage=alias):
                generator = ScriptedGenerator(
                    {key: ["FINAL ANSWER: (b) Wrong", "FINAL ANSWER: (a) Correct"]}
                )
                state = NativePath1Executor(
                    repair_stages=[alias],
                    verifiers={group: OracleAnswerVerifier()},
                ).run(_case(), generator)
                self.assertTrue(state.passed)
                self.assertEqual(len(state.attempts_for(key)), 2)
                self.assertEqual(state.verifier_calls, 2)
                second_call = [call for call in generator.calls if call["key"] == key][1]
                self.assertIsNotNone(second_call["feedback"])
                self.assertNotIn("Wrong", second_call["history"])

    def test_initial_wrong_non_idk_enters_path1_but_idk_stops(self) -> None:
        wrong = ScriptedGenerator({"init": ["FINAL ANSWER: (b) Wrong"]})
        wrong_state = NativePath1Executor().run(_case(), wrong)
        self.assertTrue(wrong_state.passed)
        self.assertEqual(wrong_state.attempts[0].evaluator.score, 0)
        self.assertIn("init", wrong_state.accepted_responses)

        idk = ScriptedGenerator({"init": ["FINAL ANSWER: (c) I don't know"]})
        idk_state = NativePath1Executor().run(_case(), idk)
        self.assertFalse(idk_state.passed)
        self.assertEqual(idk_state.stop_reason, "initial_idk")
        self.assertEqual(idk_state.model_calls, 1)

    def test_control_verifier_is_separate_from_final_evaluator(self) -> None:
        generator = ScriptedGenerator({"criteria_0": ["FINAL ANSWER: (b) Wrong"]})
        state = NativePath1Executor(
            repair_stages=["stage1"],
            verifiers={"criteria": AlwaysPassVerifier()},
        ).run(_case(), generator)
        self.assertTrue(state.passed)
        criterion_attempt = state.attempts_for("criteria_0")[0]
        self.assertEqual(criterion_attempt.evaluator.score, 0)
        self.assertTrue(criterion_attempt.gate_result.passed)

    def test_tracks_tokens_calls_and_group_repairs(self) -> None:
        generator = ScriptedGenerator(
            {"measurement": ["FINAL ANSWER: (b) Wrong", "FINAL ANSWER: (a) Correct"]}
        )
        state = NativePath1Executor(
            repair_stages=["stage3"],
            verifiers={"measurement": OracleAnswerVerifier()},
        ).run(_case(), generator)
        self.assertEqual(state.model_calls, 7)
        self.assertEqual(state.generated_tokens, 28)
        self.assertEqual(state.group_summary()["measurement"]["repairs"], 1)

    def test_can_stop_after_a_repaired_stage_for_a_conditional_probe(self) -> None:
        generator = ScriptedGenerator(
            {"bodypart_0": ["FINAL ANSWER: (b) Wrong", "FINAL ANSWER: (a) Correct"]}
        )
        state = NativePath1Executor(
            repair_stages=["stage2"],
            verifiers={"bodypart": OracleAnswerVerifier()},
        ).run(_case(), generator, stop_after_stage_key="bodypart_0")
        self.assertTrue(state.passed)
        self.assertTrue(state.completed_requested_scope)
        self.assertFalse(state.completed_full_path)
        self.assertNotIn("measurement", state.accepted_responses)

    def test_repair_feedback_can_be_overridden_without_entering_history(self) -> None:
        feedback = (
            "Re-evaluate the candidates using anatomical location, laterality, "
            "and extent before selecting an option."
        )
        generator = ScriptedGenerator(
            {"bodypart_0": ["FINAL ANSWER: (b) Wrong", "FINAL ANSWER: (a) Correct"]}
        )
        state = NativePath1Executor(
            repair_stages=["stage2"],
            verifiers={"bodypart": OracleAnswerVerifier()},
            repair_feedback_overrides={"bodypart": feedback},
        ).run(_case(), generator)

        self.assertTrue(state.passed)
        bodypart_calls = [call for call in generator.calls if call["key"] == "bodypart_0"]
        self.assertEqual(bodypart_calls[1]["feedback"], feedback)
        self.assertNotIn("Wrong", bodypart_calls[1]["history"])

    def test_repair_feedback_can_change_by_attempt_without_entering_history(self) -> None:
        first_feedback = "Return exactly one listed option."
        second_feedback = "Start over and return only one listed option."
        generator = ScriptedGenerator(
            {
                "bodypart_0": [
                    "FINAL ANSWER: (b) Wrong",
                    "FINAL ANSWER: (b) Wrong",
                    "FINAL ANSWER: (a) Correct",
                ]
            }
        )
        state = NativePath1Executor(
            repair_stages=["stage2"],
            verifiers={"bodypart": OracleAnswerVerifier()},
            max_attempts_per_stage=3,
            repair_feedback_schedules={
                "bodypart": [first_feedback, second_feedback]
            },
        ).run(_case(), generator)

        self.assertTrue(state.passed)
        bodypart_calls = [call for call in generator.calls if call["key"] == "bodypart_0"]
        self.assertEqual(bodypart_calls[1]["feedback"], first_feedback)
        self.assertEqual(bodypart_calls[2]["feedback"], second_feedback)
        self.assertNotIn("Wrong", bodypart_calls[2]["history"])

    def test_per_stage_attempt_budgets_override_the_global_default(self) -> None:
        generator = ScriptedGenerator(
            {
                "criteria_0": [
                    "FINAL ANSWER: (b) Wrong",
                    "FINAL ANSWER: (a) Correct",
                ],
                "custom_criteria": [
                    "FINAL ANSWER: (b) Wrong",
                    "FINAL ANSWER: (b) Wrong",
                    "FINAL ANSWER: (a) Correct",
                ],
            }
        )
        state = NativePath1Executor(
            repair_stages=["stage1", "stage1.5"],
            verifiers={
                "criteria": OracleAnswerVerifier(),
                "custom_criteria": OracleAnswerVerifier(),
            },
            max_attempts_per_stage=2,
            max_attempts_by_stage={"stage1.5": 3},
        ).run(_case(), generator)

        self.assertTrue(state.passed)
        self.assertEqual(len(state.attempts_for("criteria_0")), 2)
        self.assertEqual(len(state.attempts_for("custom_criteria")), 3)

    def test_normalizes_only_accepted_retry_history_and_preserves_raw_response(self) -> None:
        repaired = (
            "The highlighted anatomy is a better fit for the first candidate.\n\n"
            "FINAL ANSWER: (a) Correct"
        )
        generator = ScriptedGenerator(
            {"bodypart_0": ["FINAL ANSWER: (b) Wrong", repaired]}
        )
        state = NativePath1Executor(
            repair_stages=["stage2"],
            verifiers={"bodypart": OracleAnswerVerifier()},
            accepted_history_formatter=RepairedStageHistoryFormatter(
                "bodypart", "canonical"
            ),
        ).run(_case(), generator)

        self.assertTrue(state.passed)
        self.assertEqual(state.accepted_responses["bodypart_0"], repaired)
        measurement_call = next(
            call for call in generator.calls if call["key"] == "measurement"
        )
        self.assertIn("FINAL ANSWER: (a)", measurement_call["history"])
        self.assertNotIn(repaired, measurement_call["history"])
        self.assertEqual(generator.calls[0]["history"], [])

    def test_initial_acceptance_is_not_normalized(self) -> None:
        initial = "Reasoning that must remain.\nFINAL ANSWER: (a) Correct"
        generator = ScriptedGenerator({"init": [initial]})
        state = NativePath1Executor(
            accepted_history_formatter=RepairedStageHistoryFormatter(
                "bodypart", "canonical"
            )
        ).run(_case(), generator)

        self.assertEqual(state.accepted_history[0][2], initial)

    def test_option_text_history_format(self) -> None:
        self.assertEqual(
            normalized_selected_response(
                "Choose. Options: (a) First image, (b) Second image",
                "Reasoning. FINAL ANSWER: (b)",
                mode="option_text",
            ),
            "FINAL ANSWER: (b) Second image",
        )

    def test_history_formatter_rejects_unparseable_or_empty_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "no selected option"):
            normalized_selected_response(
                "Options: (a) First, (b) Second",
                "I cannot decide",
                mode="canonical",
            )
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            NativePath1Executor(
                accepted_history_formatter=lambda *_: "",
            ).run(_case(), ScriptedGenerator())

    def test_rejects_empty_or_unknown_feedback_overrides(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            NativePath1Executor(repair_feedback_overrides={"bodypart": " "})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            NativePath1Executor(repair_feedback_overrides={"stage9": "retry"})
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            NativePath1Executor(repair_feedback_schedules={"bodypart": []})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            NativePath1Executor(repair_feedback_schedules={"stage9": ["retry"]})
        with self.assertRaisesRegex(ValueError, "both"):
            NativePath1Executor(
                repair_feedback_overrides={"bodypart": "retry"},
                repair_feedback_schedules={"bodypart": ["retry again"]},
            )
        with self.assertRaisesRegex(ValueError, "positive integers"):
            NativePath1Executor(
                repair_stages=["stage2"],
                verifiers={"bodypart": OracleAnswerVerifier()},
                max_attempts_by_stage={"stage2": 0},
            )
        with self.assertRaisesRegex(ValueError, "corresponding repair stage"):
            NativePath1Executor(max_attempts_by_stage={"stage2": 2})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            NativePath1Executor(
                repair_stages=["stage2"],
                verifiers={"bodypart": OracleAnswerVerifier()},
                max_attempts_by_stage={"stage2": 2, "bodypart": 3},
            )

    def test_requires_a_verifier_for_every_enabled_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing verifiers"):
            NativePath1Executor(repair_stages=["stage2"])


class NativePath1ExecutorBaselineIntegrationTest(unittest.TestCase):
    workspace = Path(__file__).resolve().parents[1]
    benchmark = workspace / "physionet.org/files/chexstruct-cxreasonbench/1.0.1/CXReasonBench"
    mimic = workspace / "physionet.org/files/mimic-cxr-jpg/2.1.0/files"
    inference_root = (
        workspace
        / "outputs/medgemma4b_path1_reproduction/inference/reasoning/medgemma-4b-it"
    )
    scoring_root = (
        workspace
        / "outputs/medgemma4b_path1_reproduction/scoring/reasoning/deterministic_mcq_v1/medgemma-4b-it"
    )

    @unittest.skipUnless(
        benchmark.is_dir()
        and mimic.is_dir()
        and inference_root.is_dir()
        and scoring_root.is_dir(),
        "Downloaded benchmark and saved baseline are not available",
    )
    def test_no_repair_graph_replays_all_saved_trajectories(self) -> None:
        class ReplayGenerator:
            def __init__(self, inference: dict) -> None:
                self.records = [
                    (key.removeprefix("stage-"), value)
                    for key, value in inference.items()
                    if key.startswith("stage-")
                ]
                self.index = 0

            def generate(self, turn, accepted_history, repair_feedback=None):
                del accepted_history
                if repair_feedback is not None:
                    raise AssertionError("No-repair replay received repair feedback")
                expected_key, record = self.records[self.index]
                if turn.key != expected_key or turn.question != record["query"]:
                    raise AssertionError(
                        f"Turn mismatch: expected {expected_key}, got {turn.key}"
                    )
                self.index += 1
                return record["response"]

        loader = CXReasonBenchPath1Loader(self.benchmark, self.mimic)
        executor = NativePath1Executor()
        for task, dicom in loader.iter_case_ids(PATH1_TASKS):
            with self.subTest(task=task, dicom=dicom):
                inference = json.loads(
                    (self.inference_root / task / f"{dicom}.json").read_text(
                        encoding="utf-8"
                    )
                )
                scoring = json.loads(
                    (self.scoring_root / task / f"{dicom}.json").read_text(
                        encoding="utf-8"
                    )
                )
                generator = ReplayGenerator(inference)
                state = executor.run(loader.load_case(task, dicom), generator)
                self.assertEqual(generator.index, len(generator.records))
                self.assertEqual(
                    [attempt.stage_key for attempt in state.attempts],
                    [key for key, _ in generator.records],
                )
                for attempt in state.attempts:
                    self.assertEqual(
                        attempt.evaluator.score,
                        scoring[f"stage-{attempt.stage_key}"],
                    )
                final_attempt = state.attempts[-1]
                if final_attempt.stage_key == "final" and final_attempt.evaluator.score == 1:
                    self.assertTrue(state.passed)
                else:
                    self.assertEqual(state.failed_stage_key, final_attempt.stage_key)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from cxreason.evaluation.metrics import reasoning_depth, state_metrics
from cxreason.mock_cases.chexstruct import CorruptingCheXStructGenerator, build_chexstruct_mock_case
from cxreason.pipelines.tasks import (
    build_full_restart_task_controller,
    build_one_pass_task_controller,
    build_self_consistency_task_controller,
    build_task_controller,
)
from cxreason.tasks.registry import get_task_spec


class BaselinesAndMetricsTest(unittest.TestCase):
    def test_full_restart_regenerates_upstream_stages(self) -> None:
        task = "cardiomegaly"
        spec = get_task_spec(task)
        case = build_chexstruct_mock_case(task, row_index=1)
        generator = CorruptingCheXStructGenerator(case, spec, corrupt_stage3_once=True)

        state = build_full_restart_task_controller(task).run(case.case_id, case.task, generator)

        self.assertTrue(state.passed)
        self.assertEqual(state.attempts["stage1_criterion"], 2)
        self.assertEqual(state.attempts["stage2_anatomy"], 2)
        self.assertEqual(state.attempts["stage3_measurement"], 2)
        self.assertEqual(state.attempts["stage4_final"], 1)

    def test_local_repair_uses_fewer_calls_than_full_restart_for_stage3_failure(self) -> None:
        task = "cardiomegaly"
        spec = get_task_spec(task)
        case = build_chexstruct_mock_case(task, row_index=1)

        local = build_task_controller(task).run(
            case.case_id,
            case.task,
            CorruptingCheXStructGenerator(case, spec, corrupt_stage3_once=True),
        )
        restart = build_full_restart_task_controller(task).run(
            case.case_id,
            case.task,
            CorruptingCheXStructGenerator(case, spec, corrupt_stage3_once=True),
        )

        self.assertLess(local.model_calls, restart.model_calls)
        self.assertEqual(reasoning_depth(local, spec.stage_names), 4)
        self.assertEqual(state_metrics(local, spec.stage_names)["repaired_stages"], ["stage3_measurement"])

    def test_one_pass_fails_on_first_corrupted_stage(self) -> None:
        task = "cardiomegaly"
        spec = get_task_spec(task)
        case = build_chexstruct_mock_case(task, row_index=1)
        generator = CorruptingCheXStructGenerator(case, spec, corrupt_stage3_once=True)

        state = build_one_pass_task_controller(task).run(case.case_id, case.task, generator)

        self.assertFalse(state.passed)
        self.assertEqual(state.failed_stage, "stage3_measurement")

    def test_self_consistency_can_accept_later_candidate(self) -> None:
        task = "cardiomegaly"
        spec = get_task_spec(task)
        case = build_chexstruct_mock_case(task, row_index=1)
        generator = CorruptingCheXStructGenerator(case, spec, corrupt_stage3_once=True)

        state = build_self_consistency_task_controller(task, candidates_per_stage=3).run(
            case.case_id,
            case.task,
            generator,
        )

        self.assertTrue(state.passed)
        self.assertEqual(state.attempts["stage3_measurement"], 2)


if __name__ == "__main__":
    unittest.main()

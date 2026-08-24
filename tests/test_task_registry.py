from __future__ import annotations

import unittest

from cxreason.controller.dependency_audit import audit_accepted_chain
from cxreason.mock_cases.chexstruct import CorruptingCheXStructGenerator, build_chexstruct_mock_case
from cxreason.pipelines.tasks import build_task_controller
from cxreason.tasks.registry import TASK_REGISTRY, get_task_spec


class TaskRegistryTest(unittest.TestCase):
    def test_registers_all_cxreasonbench_tasks(self) -> None:
        self.assertEqual(
            set(TASK_REGISTRY),
            {
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
            },
        )

    def test_generic_router_runs_cardiomegaly(self) -> None:
        task = "cardiomegaly"
        case = build_chexstruct_mock_case(task, row_index=1)
        generator = CorruptingCheXStructGenerator(case, get_task_spec(task))

        state = build_task_controller(task).run(case.case_id, case.task, generator)

        self.assertTrue(state.passed)
        self.assertEqual(state.attempts["stage1_criterion"], 1)
        self.assertEqual(state.attempts["stage2_anatomy"], 1)
        self.assertEqual(state.attempts["stage3_measurement"], 2)
        self.assertEqual(state.attempts["stage4_final"], 1)

    def test_all_tasks_have_clinical_rule_final_gates(self) -> None:
        for task, spec in TASK_REGISTRY.items():
            with self.subTest(task=task):
                self.assertEqual(spec.verification_levels["stage4_final"], "clinical_rule")

    def test_all_task_final_gates_detect_flipped_decision(self) -> None:
        for task in TASK_REGISTRY:
            with self.subTest(task=task):
                spec = get_task_spec(task)
                case = build_chexstruct_mock_case(task, row_index=0)
                generator = CorruptingCheXStructGenerator(
                    case,
                    spec,
                    corrupt_stage3_once=False,
                    corrupt_stage4_once=True,
                )

                state = build_task_controller(task).run(case.case_id, case.task, generator)

                self.assertTrue(state.passed)
                self.assertFalse(state.gate_results["stage4_final"][0].passed)
                self.assertTrue(state.gate_results["stage4_final"][1].passed)

    def test_all_task_stage3_gates_detect_mock_corruption(self) -> None:
        for task in TASK_REGISTRY:
            with self.subTest(task=task):
                spec = get_task_spec(task)
                case = build_chexstruct_mock_case(task, row_index=0)
                generator = CorruptingCheXStructGenerator(case, spec, corrupt_stage3_once=True)

                state = build_task_controller(task).run(case.case_id, case.task, generator)

                self.assertTrue(state.passed)
                self.assertFalse(state.gate_results["stage3_measurement"][0].passed)
                self.assertTrue(state.gate_results["stage3_measurement"][1].passed)

    def test_dependency_audit_rechecks_accepted_chain(self) -> None:
        task = "cardiomegaly"
        spec = get_task_spec(task)
        case = build_chexstruct_mock_case(task, row_index=1)
        generator = CorruptingCheXStructGenerator(case, spec, corrupt_stage3_once=False)
        state = build_task_controller(task).run(case.case_id, case.task, generator)

        self.assertTrue(audit_accepted_chain(state, spec).passed)

        state.accepted_outputs["stage3_measurement"]["ctr"] = 9.0

        result = audit_accepted_chain(state, spec)
        self.assertFalse(result.passed)
        self.assertEqual(result.metadata["stage"], "stage3_measurement")


if __name__ == "__main__":
    unittest.main()

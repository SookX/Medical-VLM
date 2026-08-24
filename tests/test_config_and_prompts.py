from __future__ import annotations

import unittest

from cxreason.config import load_config
from cxreason.prompting.tasks import build_all_task_prompt_templates, build_task_stage_prompt
from cxreason.tasks.registry import TASK_REGISTRY


class ConfigAndPromptsTest(unittest.TestCase):
    def test_load_config_extends_default(self) -> None:
        config = load_config("configs/medgemma.yaml")

        self.assertEqual(config["controller"]["max_attempts_per_stage"], 3)
        self.assertEqual(config["model"]["model_id"], "google/medgemma-4b-it")

    def test_all_task_prompts_render_all_stages(self) -> None:
        prompts = build_all_task_prompt_templates()

        self.assertEqual(set(prompts), set(TASK_REGISTRY))
        for task, stage_prompts in prompts.items():
            self.assertEqual(set(stage_prompts), set(TASK_REGISTRY[task].stage_names))

    def test_stage2_prompt_contains_task_specific_fields(self) -> None:
        prompt = build_task_stage_prompt("rotation", "stage2_anatomy", {})

        self.assertIn("medial_end_right_clavicle", prompt)
        self.assertIn("midline_points", prompt)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.evaluation.logging import write_json
from cxreason.prompting.tasks import build_all_task_prompt_templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Render all task prompt templates.")
    parser.add_argument("--output", default="outputs/task_prompts.json")
    args = parser.parse_args()

    prompts = build_all_task_prompt_templates()
    write_json(args.output, prompts)
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()

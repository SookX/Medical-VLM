from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxreason.evaluation.coverage import render_markdown_coverage, task_coverage_rows
from cxreason.evaluation.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Write task verifier coverage report.")
    parser.add_argument("--markdown", default="outputs/task_coverage.md")
    parser.add_argument("--json", default="outputs/task_coverage.json")
    args = parser.parse_args()

    rows = task_coverage_rows()
    markdown = render_markdown_coverage(rows)

    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    write_json(args.json, rows)

    print(f"markdown: {markdown_path}")
    print(f"json: {args.json}")


if __name__ == "__main__":
    main()

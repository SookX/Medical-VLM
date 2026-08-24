# Experiment Skeleton

The current image-free experiment skeleton includes:

- local repair controller
- one-pass / greedy structured baseline
- self-consistency baseline
- full-chain restart baseline
- reusable metrics
- corruption evaluation
- task coverage reporting
- oracle gate interface
- YAML-driven mock runs with environment metadata
- all-task prompt templates
- MedGemma text-only dry-run path

## Metrics

Reusable metric helpers live in:

```text
cxreason/evaluation/metrics.py
```

They report:

- completion
- reasoning depth
- first-attempt stage pass/fail
- post-repair stage pass/fail
- repaired stages
- repair success
- false-repair rate
- model calls
- verifier calls
- dependency-audit pass rate

## Full-Chain Restart

The full-restart baseline lives in:

```text
cxreason/baselines/full_restart.py
```

When a stage fails verification, it discards the current trajectory and restarts
from Stage 1. This contrasts with local repair, which retries only the failed
stage.

Compare both systems:

```powershell
python scripts\compare_local_vs_full_restart.py --modes stage3 stage4 --count 2 --output-dir outputs\local_vs_full_restart_all_tasks_smoke
```

Current smoke result:

```text
Stage 3 failure:
  local repair mean calls: 5
  full restart mean calls: 7

Stage 4 failure:
  local repair mean calls: 5
  full restart mean calls: 8
```

Both completed all smoke cases. The difference is compute: full restart
regenerates already accepted upstream stages.

## Mock Baselines

Run all current mock baselines from CLI flags:

```powershell
python scripts\run_mock_baselines.py --modes stage3 stage4 --count 2 --output-dir outputs\mock_baselines
```

Run from YAML config:

```powershell
python scripts\run_mock_baselines.py --config configs\mock_baselines.yaml
```

Current systems:

- `one_pass`: one generation per stage, no retry.
- `self_consistency`: multiple candidates per stage, accept first verified candidate.
- `local_repair`: retry only the failed stage.
- `full_restart`: restart the full Path-1 chain after any failed stage.

Each run writes `attempts.jsonl`, `runs.jsonl`, `summary.json`,
`system_summary.json`, and `environment.json`.

## Task Coverage

Generate coverage reports:

```powershell
python scripts\report_task_coverage.py --markdown outputs\task_coverage.md --json outputs\task_coverage.json
```

The report lists verifier strength and gates per task/stage.

## Oracle Gate

The oracle interface lives in:

```text
cxreason/gates/oracle.py
cxreason/pipelines/oracle.py
```

The oracle gate returns only:

```text
PASS
REPAIR
```

It does not expose the hidden answer in `reason`, `metadata`, or repair
feedback. The current `MockAnswerScorer` is for development only; later it can
be replaced by the official CXReasonBench scorer.

Run the mock oracle experiment:

```powershell
python scripts\run_mock_oracle_eval.py --tasks cardiomegaly --count 2 --output-dir outputs\mock_oracle_smoke
```

## Prompts

All-task prompt templates are generated from the task registry:

```powershell
python scripts\render_task_prompts.py --output outputs\task_prompts.json
```

Render a MedGemma text-only dry run without loading the model:

```powershell
python scripts\run_medgemma_text_dry_run.py --task cardiomegaly --row-index 1
```

Actually call MedGemma in text-only mode:

```powershell
python scripts\run_medgemma_text_dry_run.py --task cardiomegaly --row-index 1 --call-model
```

The text-only mode includes development structured context and is for JSON
compliance/repair-prompt testing only. It is not a visual benchmark result.

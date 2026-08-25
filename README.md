# Medical-VLM

Training-free stage-gated diagnostic reasoning experiments for CXReasonBench
Path 1.

## Current Runnable Slice

The repository can run a mock cardiomegaly Path-1 pipeline without the restricted
CXReasonBench dataset or real X-ray images. It uses CheXStruct CTR fields as
structured mock evidence.

Run one local-repair trace:

```powershell
python scripts\run_cardiomegaly_mock_pipeline.py --row-index 1
```

Run a 100-case CheXStruct mock sweep:

```powershell
python scripts\run_cardiomegaly_mock_pipeline.py --row-index 0 --count 100
```

Run tests:

```powershell
python -m unittest discover -s tests
```

Run the same controller with MedGemma on an image:

```powershell
python scripts\run_cardiomegaly_medgemma_pipeline.py --image path\to\chest_xray.png
```

This command exercises the model-backed generation path. Its verification only
becomes meaningful when the image is a real frontal CXR and the pipeline has
trusted anatomy/measurement evidence or official CXReasonBench stage references
to compare against.

## What The Mock Pipeline Does

For cardiomegaly, the mock Path-1 stages are:

1. Select the diagnostic criterion: cardiothoracic ratio (CTR).
2. Ground the required anatomy: heart width and thoracic/lung width.
3. Compute the measurement: `CTR = heart_width / lung_width`.
4. Apply the final rule: cardiomegaly yes/no from CTR and view position.

The demo intentionally corrupts Stage 3 once. The controller rejects the bad CTR,
repairs only Stage 3, preserves the accepted Stage 1 and Stage 2 outputs, and
then continues to Stage 4.

The cardiomegaly thresholds are CheXStruct-compatible mock thresholds inferred
from the public CSV label boundary: PA `0.495`, AP `0.545`. Treat these as
development scaffolding until the official CXReasonBench cases and criteria are
plugged in.

The MedGemma stage generator asks for one JSON object per stage and repair
feedback never includes benchmark answers. The final-rule verifier derives the
CTR threshold from trusted view-position context, not from model-provided text.

## Task Router

The 12 CXReasonBench tasks are registered in a deterministic router at
`cxreason/tasks/registry.py`. This is intentionally not a learned MoE:

```text
task name -> task expert spec -> stage-specific gates
```

Run any registered task through the CheXStruct mock pipeline:

```powershell
python scripts\run_mock_task_pipeline.py --task rotation --row-index 0
```

See `docs/task_router.md` for the Stage 2 schemas.
See `docs/clinical_rules.md` for the current deterministic Stage 4 rules.

## Corruption Evaluation

Run controlled corruptions across the mock pipeline:

```powershell
python scripts\run_corruption_eval.py --tasks cardiomegaly --count 3 --output-dir outputs\corruption_eval_smoke
python scripts\run_corruption_eval.py --count 10 --output-dir outputs\corruption_eval_all_tasks
```

The evaluator writes `attempts.jsonl`, `runs.jsonl`, and `summary.json`.
See `docs/corruption_eval.md` for details.

The corruption evaluator also records a controller-level dependency audit, which
replays the final accepted chain against the task spec after repair.

## Baselines And Metrics

Run all mock baselines:

```powershell
python scripts\run_mock_baselines.py --config configs\mock_baselines.yaml
```

Compare local repair against full-chain restart:

```powershell
python scripts\compare_local_vs_full_restart.py --modes stage3 stage4 --count 2 --output-dir outputs\local_vs_full_restart_all_tasks_smoke
```

Generate a task coverage report:

```powershell
python scripts\report_task_coverage.py --markdown outputs\task_coverage.md --json outputs\task_coverage.json
```

See `docs/experiment_skeleton.md` for the reusable metrics, baseline, coverage,
and oracle-gate interfaces.
See `docs/image_free_implementation_summary.md` for the full image-free
implementation summary.

Render all task prompts:

```powershell
python scripts\render_task_prompts.py --output outputs\task_prompts.json
```

Run mock oracle local repair:

```powershell
python scripts\run_mock_oracle_eval.py --tasks cardiomegaly --count 2 --output-dir outputs\mock_oracle_smoke
```

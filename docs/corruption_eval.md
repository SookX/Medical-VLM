# Corruption Evaluation

Controlled corruption evaluation is the first validation experiment that does
not require the restricted CXReasonBench case package.

The script intentionally corrupts one stage on the first attempt and then lets
the mock generator return the correct CheXStruct-backed output on repair.

```powershell
python scripts\run_corruption_eval.py --tasks cardiomegaly --count 3 --output-dir outputs\corruption_eval_smoke
```

Run all registered tasks:

```powershell
python scripts\run_corruption_eval.py --count 10 --output-dir outputs\corruption_eval_all_tasks
```

## Outputs

Each run writes:

- `attempts.jsonl`: one record per stage attempt, including output, gate result,
  verification level, accepted flag, and accounting.
- `runs.jsonl`: one record per corruption run.
- `summary.json`: grouped detection/completion metrics.

## Metrics

The summary reports:

- `detection_rate`: fraction of corrupted first attempts rejected by the gate.
- `completion_rate`: fraction of cases that complete after allowed repair.
- `repair_success_rate`: fraction of detected corruptions that complete after repair.
- `false_repair_rate`: fraction of clean cases where a first attempt was wrongly rejected.
- `dependency_audit_pass_rate`: fraction of completed accepted chains that pass controller-level re-verification.
- `mean_model_calls`
- `mean_verifier_calls`

## Current Smoke Result

After adding deterministic final-rule gates for all 12 tasks, the all-task smoke
run used 2 rows per task and produced 120 runs. After adding Stage 3 sanity
gates, the result is:

```text
clean: 24 cases, 0 false repairs
stage1 criterion corruptions: 24/24 detected
stage2 anatomy corruptions: 24/24 detected
stage3 arithmetic corruptions: 12/12 detected
stage3 sanity corruptions: 12/12 detected
stage4 clinical-rule corruptions: 24/24 detected
dependency audit: 120/120 passed
```

The Stage 3 sanity gates catch out-of-domain structured measurements such as
impossible angles, ratios outside `[0, 1]`, impossible rib counts, invalid binary
regional labels, and invalid trachea-direction choices. These are still weaker
than independent visual verification: they validate plausibility and internal
format, not whether the image evidence truly supports the value.

An isolated Stage 4 corruption run over 10 rows per task produced:

```text
stage4 clinical-rule corruptions: 120/120 detected
completion: 120/120
false repairs: 0
```

The dependency audit is implemented as a controller-level re-verification pass.
It does not introduce new medical rules; it checks that the final accepted chain
still satisfies each stage gate when replayed with the appropriate accepted
upstream context.

# Medical-VLM

Training-free stage-gated diagnostic reasoning experiments for CXReasonBench
Path 1.

For the consolidated implementation status, experimental results, accepted and
rejected components, reproduction commands, and remaining work, see
[`PROJECT_STATUS_SUMMARY.md`](PROJECT_STATUS_SUMMARY.md).

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

Run one full, narrated image-free pipeline example:

```powershell
python scripts\run_full_pipeline_example.py --task cardiomegaly --row-index 1 --mode stage3
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

## Selective Path-1 consensus pipeline

The repository includes native CXReasonBench Path-1 loading, MedGemma execution,
stage-local repair, deterministic measurement geometry, confidence consensus,
and the original-metric layout adapter. The frozen three-measurement development
configuration is `configs/path1_three_measurement_consensus_v1.json`.

Run the focused validation with:

```powershell
python scripts\audit_path1_measurement_consensus_v2.py --task aortic_knob_enlargement --version v3
python scripts\audit_path1_measurement_consensus_v2.py --task carina_angle --version v2
python scripts\audit_path1_measurement_consensus_v2.py --task descending_aorta_enlargement --version v3
python scripts\verify_path1_three_measurement_consensus_v1.py
```

The benchmark and model artifacts are intentionally not committed. See
`docs/path1_12_task_completion_program.md` and `docs/PATH1_APPENDIX.md` for the
claim boundary, ablations, and required local artifact layout.

Render all task prompts:

```powershell
python scripts\render_task_prompts.py --output outputs\task_prompts.json
```

Run mock oracle local repair:

```powershell
python scripts\run_mock_oracle_eval.py --tasks cardiomegaly --count 2 --output-dir outputs\mock_oracle_smoke
```

## Path 1 study records

The current implementation/ablation checklist is in
`PATH1_IMPLEMENTATION_AND_ABLATION_PLAN.md`. The living paper appendix,
including calibration cohorts, locked policies, negative results, compute
controls, artifact locations, and the final appendix checklist, is in
`docs/PATH1_APPENDIX.md`.

The projection upstream calibration, transfer audit, and cumulative ablation
can be reproduced with:

```powershell
python scripts\calibrate_path1_projection_bodypart.py
python scripts\audit_path1_projection_upstream_verifiers.py
python scripts\run_path1_projection_cumulative_ablation.py
python scripts\run_path1_projection_repair_policy_ablation.py
python scripts\run_path1_projection_selected_policy.py
python scripts\run_path1_projection_stage2_repair_policy.py
python scripts\run_path1_projection_stage2_strict_confirmation.py
python scripts\run_path1_projection_full_restart_control.py
python scripts\verify_path1_projection_full_restart_control.py
python scripts\verify_path1_projection_frozen_config.py
python scripts\run_path1_inclusion_unified.py
python scripts\verify_path1_inclusion_unified.py
python scripts\run_path1_inclusion_full_restart_control.py
python scripts\verify_path1_inclusion_full_restart_control.py
python scripts\verify_path1_inclusion_frozen_config.py
python scripts\prepare_path1_cardiomegaly_stage2_calibration.py
python scripts\calibrate_path1_cardiomegaly_stage2.py
python scripts\audit_path1_locked_independent_cardiomegaly_stage2.py
python scripts\verify_path1_cardiomegaly_stage2.py
python scripts\prepare_path1_mediastinal_stage2_calibration.py
python scripts\calibrate_path1_mediastinal_stage2.py
python scripts\audit_path1_locked_independent_mediastinal_stage2.py
python scripts\verify_path1_mediastinal_stage2.py
python scripts\audit_path1_task_support.py
python scripts\verify_path1_task_support.py
python scripts\verify_path1_remaining_stage2.py
python scripts\verify_path1_inclusion_unified_b1.py
python scripts\verify_path1_inclusion_unified_b2.py
python scripts\verify_path1_projection_b1.py
python scripts\verify_path1_projection_b2.py
python scripts\verify_path1_end_to_end_readiness.py
python scripts\evaluate_path1_official_metrics.py
```

See `docs/path1_projection_upstream_pipeline.md` for the locked thresholds,
zero-false-repair Stage-1/1.5/2 transfer audit, full 100-case cumulative run,
retry-policy and call-matched blind ablation, selected-policy end-to-end
confirmation, the Stage-2 prompt/retry pilot, and remaining projection work.
The strict Stage-2 end-to-end confirmation is also included; it advances one
additional case to Stage 3 without regressing any trajectory.
The 144-call full-chain restart control and permanent artifact verifier test
whether an equal explicit-call budget explains the selected local-repair gain.
The final projection v1 prompts, thresholds, budgets, code hashes, artifacts,
compute rules, and environment are frozen in
`configs/path1_projection_frozen_v1.json`; see
`docs/path1_projection_frozen_configuration.md`.
The frozen inclusion v1 policy improves Completion from 10% to 16% and mean
Depth from 0.85 to 1.01 over all 100 inclusion cases, with seven paired depth
wins and no losses. Its 38-call full-restart control reaches 11% Completion
and 0.87 mean Depth. The configuration and claim boundary are documented in
`docs/path1_inclusion_frozen_configuration.md`.
Its unified 38-call blind-local-retry control remains at 10% Completion and
0.85 mean Depth, while frozen B4 reaches 16% and 1.01 with no depth loss. See
`docs/path1_inclusion_unified_b1.md`.
Its sampled 38-call, stage-allocation-matched B2 control falls to 9%
Completion and 0.81 mean Depth despite using only 27 more generated tokens
than B4. See `docs/path1_inclusion_unified_b2.md`.
Projection's 144-call blind B1 remains at mean Depth 0.00 and sampled B2
reaches 0.01, while frozen routed B4 reaches 0.13. See
`docs/path1_projection_compute_controls.md`.
The all-task support gate is in `configs/path1_task_support_v5.json`; its
expanded 60-cell audit and final-run blockers are documented in
`docs/path1_task_support_matrix.md`.
The original-metric adapter now evaluates frozen task replacements in the
native CXReasonBench layout over all 1,200 cases. The current Inclusion plus
Projection overlay improves local-baseline Completion from 1.73 to 2.17,
Depth from 0.36 to 0.38, and Alignment from 43.53 to 48.78. See
`docs/path1_official_metric_adapter.md`; unsupported tasks remain native
baseline pass-throughs, so this is not yet the final all-task result.
The first Cardiomegaly Stage-2 candidate is retained as a negative result: it
passed external validation with zero wrong conclusive mappings but caused
16/166 false repairs in the locked benchmark transfer audit. No MedGemma
repair run was allowed. See `docs/path1_cardiomegaly_stage2_verifier.md`.
The first Mediastinal Widening Stage-2 candidate is also retained as a
negative result. It passed external validation with zero wrong conclusive
mappings but produced 3/166 false repairs after locked transfer, so no
MedGemma repair run was allowed. See
`docs/path1_mediastinal_stage2_verifier.md`.
The remaining eight Stage-2 candidates are now closed: two failed external
validation, three failed locked benchmark transfer, and three safely routed 20
MedGemma retries but corrected 0/20. All are rejected and pass through natively.
See `docs/path1_remaining_stage2_verifiers.md`.
The frozen 1,200-case supported-task overlay now has a hash-locked preflight
covering all baseline cases, both replacement cohorts, and all six B1/B2/B3
control verifications. See `docs/path1_supported_final_run.md` for the single
copy-paste final command. Ten tasks remain native pass-throughs.

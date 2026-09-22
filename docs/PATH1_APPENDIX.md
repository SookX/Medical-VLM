# Path 1 Experimental Appendix

This is the living appendix for the Path 1 MedGemma reproduction and verifier-
guided repair study. It records enough detail to reproduce each reported
component and distinguishes accepted, rejected, and diagnostic-only results.
It will be frozen together with the final experiment configuration.

## A. Reproducibility snapshot

| Item | Locked value or policy |
|---|---|
| Model | `google/medgemma-4b-it` |
| Model revision | `290cda5eeccbee130f987c4ad74a59ae6f196408` |
| Image size | 1024 |
| Decoding | Greedy, seed 42 |
| Selected projection attempt policy | 2 total attempts at Stages 1/3/4; 3 at Stages 1.5/2; strict feedback only at Stage 2 |
| Frozen projection configuration | `path1_projection_frozen_v1`, SHA-256 `4c7c6658fb7167701a1f35461f9f31099ea186e22ba1e59a1f119ab21accc16c` |
| Frozen inclusion configuration | `path1_inclusion_frozen_v1`, SHA-256 `f979b3f38e3381d1c2ea02b8c2fbdd24ba325d80e4afcf138ef4835c8a4f7cae` |
| Frozen Carina configuration | `path1_carina_selector_frozen_v2`, SHA-256 `a7d3644fd76a5a190bb7c3d6f030406bb66773199b0a629411dc1285a38bc80e` |
| All-task downstream support gate | `path1_task_support_v6`; Carina, Inclusion, and Projection final-ready; nine Stage-2 policies rejected; all-task Stage-2 intervention disabled |
| Benchmark | CXReasonBench 1.0.1, Path 1 |
| Benchmark size | 1,200 cases, 100 per task |
| Independent segmenter | TorchXRayVision ChestX-Det PSPNet 1.5.4 |
| Segmenter weight SHA-256 | `019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc` |
| Current regression suite | 191 tests plus 2,441 subtests |

The golden no-repair baseline remains protected by
`configs/baselines/medgemma4b_path1_golden.json`. Every experiment records its
configuration, model revision, policy hash, source hashes, generation source,
token count, runtime, and peak GPU memory where applicable.

## B. Leakage and split controls

- Practical verifiers never receive benchmark reference answers at runtime.
- Evaluation output cannot affect controller routing.
- External splits are patient-disjoint and conservatively exclude potential
  ChestX-Det training overlap.
- Thresholds and uncertainty policies are selected on calibration only and
  serialized before validation is evaluated.
- Sealed external test images are not downloaded, predicted, or scored.
- Benchmark safety audits occur only after external policy locking.
- A candidate that fails its declared validation gate is retained as a
  rejected result; it is not retuned on validation.
- Pipeline-derived CheXStruct labels are described as such and are not called
  radiologist ground truth.

## C. Component status and evidence

| Component | Status | Development/transfer evidence | Repair evidence |
|---|---|---|---|
| Stage-4 rule consistency, all tasks | Engineering validated | 968/1,200 conclusive, zero mapping disagreements | 19-case reached-final cohort: 1/2 repairs; Completion 89.47% to 94.74% |
| Inclusion Stage-2 grounding | Accepted engineering component | 119/150 conclusive and correct after external lock | 50-case cohort: Completion 20% to 28%, 5/32 repairs, zero damage |
| Inclusion Stage-3 geometry | Accepted engineering component | 100/100 references protected; 266/400 wrong alternatives detected | 14-case cohort: Completion 71.43% to 78.57%, 1/2 repairs, zero damage |
| Cardiomegaly Stage-3 CTR | **Rejected** | 2/71 validation false repairs; 4.23% wrong-alternative detection | Not run |
| Cardiomegaly Stage-2 grounding | **Rejected** | External validation: 749/799 conclusive/correct and 0 wrong; locked transfer: 16/166 false repairs | Not run |
| Mediastinal Widening Stage-2 grounding | **Rejected** | External validation: 696/799 conclusive/correct and 0 wrong; locked transfer: 3/166 false repairs | Not run |
| Projection Stage-3 overlap | Accepted for routing | 100/100 references protected; 766/2,400 wrong pairs detected | Not runnable: 0 native trajectories reach Stage 3 |
| Projection Stage-1.5 applicability | Accepted engineering component | 100/100 references protected; all 22 reached baseline errors flagged | Two generic retries corrected 12/22; selected full run doubled mean Depth from 0.06 to 0.12 |
| Projection Stage-1 criterion | Accepted protocol-semantic component | 150/150 references protected; 600/600 wrong alternatives detected | Exploratory policy ablation: one generic retry 5/78; two generic retries 7/78 |
| Projection Stage-2 grounding | Verifier accepted; strict repair selected as exploratory engineering policy | External validation: 640/646 correct, 0 wrong, 6 abstained; transfer: 103/166 conclusive, 0 false repairs | Strict formatting corrected 1/8 and raised full-run mean Depth from 0.120 to 0.130 with zero regressions |
| Projection configuration v1 | **Frozen and verified** | 13 source/artifact locks plus exact environment | 100 case hashes, 144 repair calls, and 144 restart records verified |
| Inclusion configuration v1 | **Frozen and verified** | 18 source/artifact locks plus exact environment | Completion 10% to 16%; mean Depth 0.85 to 1.01; 38 repair and restart records verified |
| All-task support matrix | **Audited safety gate** | 12 tasks and 60 task-stage cells verified | 18/60 cells enabled; inclusion and projection final-ready |
| Selective protocol Stage 1/1.5 | **Verified all-task component** | 2,500 decisions checked; 1,274 preserved and 1,226 repaired | Mean Depth 0.341 to 1.068; Completion 17 to 19; 863 depth wins, 0 losses |
| Inspiration focused Stage-3 prompt | **Rejected** | Frozen 60-case certified-anatomy cohort; one call per arm | 12/60 native vs 12/60 focused; 7 gains and 7 losses |
| Exact-prefix composed overlay | **Verified, metric trade-off** | 169/300 downstream trajectories compatible; 1,200 composition decisions verified | Completion 17 to 34; mean Depth 0.341 to 1.137; zero losses; official Alignment 43.53 to 29.64 |

These are component-level engineering results. They are not the final full-
benchmark comparison, and the small frozen repair cohorts should not be used
as precise effect estimates.

The selected projection retry policy was also confirmed across all 100 cases.
It reproduced six correct second Stage-1.5 repairs, increased Stage-2 reach
from 6 to 12 cases, improved depth for six cases with no regressions, and
doubled mean Depth from 0.060 to 0.120. Completion remained 0/100 because all
eight routed Stage-2 retries remained wrong.

The subsequent frozen Stage-2 error-cohort ablation compared one generic
retry, two generic retries, two structured retries, and up to two strict-
format retries. Only strict formatting converted a case (1/8; Wilson 95% CI
2.24%-47.09%); all prompt-matched blind controls completed 0/8. The paired
strict-versus-generic result is `p=1.0`, so the policy remains exploratory.
Its full 100-case propagation improved that case from Depth 1 to 2, changed no
other trajectory, and produced no regression. Stage-3 reach increased from
four to five, but the new path failed both Stage-3 attempts and Completion
remained 0/100. Strict Stage-2 feedback is selected for the engineering
configuration under the predeclared no-damage gate, while the one-case effect
remains explicitly exploratory.

## D. Locked inclusion policies

### D.1 Stage-2 candidate matching

| Parameter | Value |
|---|---:|
| Minimum Dice | 0.75 |
| Maximum Dice for special/no-match option | 0.35 |
| Minimum best-versus-second Dice margin | 0.17 |

### D.2 Stage-3 visible margins

| Region | Minimum margin |
|---|---:|
| Right apex | 0.065 |
| Left apex | 0.067 |
| Right lateral edge | 0.026 |
| Left lateral edge | 0.022 |
| Right bottom | 0.094 |
| Left bottom | 0.079 |

The inclusion calibration uses 455 patients, validation uses 112, and 137
external test patients remain sealed.

## E. Compute-matched controls completed so far

On the frozen 50-case inclusion Stage-2 cohort:

| System | Calls allocated to intervention | Completion | Mean Depth | Damage |
|---|---:|---:|---:|---:|
| Native `B0` | 0 | 20% | 1.70 | 0 |
| Blind retry `B1` | 33 | 18% | 1.62 | 4 |
| Sampled self-consistency `B2` | 33 | 18% | — | 3 |
| Full-chain restart `B3` | 33 | 20% | — | 0 |
| Routed verifier repair `B4` prototype | 33 | 26% | 1.90 | 0 |
| Externally locked Stage-2 verifier | 32 | 28% | 1.96 | 0 |

The final paper table will recompute comparable metrics from a single frozen
configuration rather than mixing prototype and locked-policy rows.

On the frozen unified 100-case inclusion configuration:

| System | Explicit intervention calls | Completion | Mean Depth |
|---|---:|---:|---:|
| Native `B0` | 0 | 10% | 0.85 |
| Blind local `B1` | 38 | 10% | 0.85 |
| Stage 2 only | 32 | 14% | 0.98 |
| Stages 2-3 | 36 | 15% | 1.00 |
| Stages 2-4 local repair | 38 | **16%** | **1.01** |
| Call-matched full restart | 38 | 11% | 0.87 |

The full local policy improves seven depths and damages none relative to native
(`p=0.015625`). Against call-matched restart, it wins eight paired depths and
loses one (`p=0.039062`). Local repair opens 18 downstream calls, so its total
incremental count is 56 even though the primary explicit-call budget is 38.
The stage-allocation-matched blind B1 control has five depth wins and seven
losses, leaving aggregate Completion and Depth unchanged; it also spends 6,632
intervention tokens versus B4's 2,733.
The stage-allocation-matched sampled B2 control reaches 9% Completion and 0.81
mean Depth, with zero depth wins and two losses. Its 2,760 intervention tokens
are within 27 of B4, strengthening the conclusion that the B4 gain is not a
sampling or token-budget effect.

On the frozen projection upstream error cohorts:

| Stage and policy | Routed correct | Call-matched blind correct | Retry calls | Paired routed-vs-blind p |
|---|---:|---:|---:|---:|
| Stage 1, one generic retry | 5/78 | 5/78 | 78 | 1.0 |
| Stage 1, two generic retries | 7/78 | 5/78 | 151 | 0.5 |
| Stage 1, two strict retries | 6/78 | 5/78 | 151 | 1.0 |
| Stage 1.5, one generic retry | 6/22 | 6/22 | 22 | 1.0 |
| Stage 1.5, two generic retries | 12/22 | 10/22 | 38 | 0.5 |
| Stage 1.5, two strict retries | 4/22 | 4/22 | 41 | 1.0 |
| Stage 2, one generic retry | 0/8 | 0/8 | 8 | 1.0 |
| Stage 2, two generic retries | 0/8 | 0/8 | 16 | 1.0 |
| Stage 2, two structured retries | 0/8 | 0/8 | 16 | 1.0 |
| Stage 2, up to two strict retries | 1/8 | 0/8 | 15 | 1.0 |

The Stage-1.5 two-generic policy added six paired corrections over one retry
(`p=0.03125`) using 16 additional calls. Strict output formatting was worse
than two generic retries (`p=0.007812`). These are exploratory engineering
results on baseline-error cohorts, not held-out effect estimates.

The projection full-chain restart control matched the selected local policy's
144 explicit repair calls with 144 sampled restart calls. Local repair reached
mean Depth 0.130 and improved 12 cases over native `B0`; restart reached mean
Depth 0.010 and improved one. In the paired local-versus-restart comparison,
local repair won 12 cases, restart won one, and 87 tied (`p=0.003418`). Both
systems remained at 0/100 Completion. Restart generated 8,983 intervention
tokens versus 7,716 for local repair. The control is intervention-call matched:
local repair also induced 25 newly reachable downstream calls.

Two additional 144-call controls operate at `criteria_0`, the only stage
uniformly reached by native Projection. Blind B1 selected 72 cases for two
retries each and remained at mean Depth 0.00. Sampled B2 selected 48 cases for
three votes each and reached mean Depth 0.01 through one depth gain. B1 and B2
used 10,089 and 12,207 intervention tokens respectively, versus 6,343 retry
tokens for B4. B4 still obtained 12 depth gains and mean Depth 0.13. Exact
per-stage allocation matching is impossible without importing B4's dynamic
routing decisions, so these are explicitly total-call-matched controls.

### E.1 Original-metric 1,200-case supported-task overlay

The native-layout adapter was regression-tested by exporting the untouched
local `B0` and reproducing all seven aggregate metric values exactly. Frozen
Inclusion and Projection policies were then overlaid as complete 100-case task
cohorts while the remaining ten tasks stayed on native `B0`.

| Configuration | Completion (refined) | Depth (refined) | Consistency | Alignment (refined) |
|---|---:|---:|---:|---:|
| Local `B0` | 1.73 (1.32) | 0.36 (0.35) | 32.06 | 43.53 (32.74) |
| Inclusion only | 2.17 (1.75) | 0.37 (0.36) | 32.06 | 48.78 (37.99) |
| Projection only | 1.73 (1.32) | 0.37 (0.36) | 32.06 | 43.53 (32.74) |
| Inclusion + Projection | **2.17 (1.75)** | **0.38 (0.38)** | 32.06 | **48.78 (37.99)** |

The adapter enforces the original evaluator's correct-prefix assumption.
Downstream turns reached only through independent-verifier routing receive no
paper-metric credit after a benchmark-incorrect prerequisite. This affected
five Inclusion cases/eight downstream stages and four Projection cases/seven
downstream stages. Source controller artifacts remain unchanged. The combined
machine-readable result is
`outputs/path1_official_metrics_supported_v1/metrics.json`; full methodology
and commands are in `docs/path1_official_metric_adapter.md`.

This is a supported-task overlay, not the final all-task intervention. It uses
the deterministic multiple-choice scorer because the published Gemini 2.0
Flash answer-equivalence scorer is retired.

## F. Negative and failed results

Negative results are part of the proposed contribution and remain in the
appendix:

- Direct point-ratio transfer to the inclusion overlay caused 48/100 false
  repairs and was rejected.
- Independent cardiothoracic mask widths caused 2/71 external validation false
  repairs and detected only 4.23% of wrong alternatives; the candidate was
  rejected.
- Independent Cardiomegaly Stage-2 Heart/line grounding passed external
  validation but produced 16/166 false repairs after locked transfer. No model
  repair experiment was run.
- Independent Mediastinal Widening Stage-2 grounding passed external
  validation but produced 3/166 false repairs after locked transfer. No model
  repair experiment was run.
- Aortic-Knob Enlargement and Trachea Deviation each produced one wrong
  conclusive external-validation mapping; transfer was not run.
- Rotation, Descending-Aorta Enlargement, and Descending-Aorta Tortuosity
  produced 33/166, 9/166, and 5/150 reference false repairs after locked
  transfer; generation was not run.
- Ascending-Aorta Enlargement, Carina Angle, and Inspiration protected every
  correct reference but converted 0/20 targeted MedGemma retries. Completion
  and depth were unchanged.
- Concise Stage-2 feedback reduced retry tokens but lost all completion gain.
- Letter-only and letter-plus-option normalized histories underperformed full
  accepted repair responses.
- Blind retry, sampled plurality, and complete-chain restart did not explain
  the routed inclusion Stage-2 gain.
- The projection cumulative run safely opened six paths at Stage 1.5 but did
  not convert any of four Stage-2 retries or the one Stage-3 retry; completion
  remained 0/22 while mean depth rose from 0.000 to 0.273.
- The full projection run converted 5/78 Stage-1 retries and 6/27 Stage-1.5
  retries, but none of the newly opened Stage-1 paths survived Stage 1.5.
  Completion remained 0/100 and mean depth reached only 0.060.
- Strict-format projection repairs did not improve Stage 1 and reduced
  Stage-1.5 repair conversion from 12/22 with two generic retries to 4/22.
- The selected projection policy doubled mean Depth and improved six cases,
  but still produced 0/100 Completion; the bottleneck moved to Stage 2.
- Generic and structured Stage-2 repair policies corrected 0/8 frozen errors.
  Strict formatting corrected 1/8, but the cohort is too small and the paired
  result (`p=1.0`) is insufficient for an efficacy claim.
- The 144-call projection full-chain restart control reached mean Depth 0.010,
  well below selected local repair at 0.130, and completed no path.

## G. Required final appendix tables

The following tables will be generated after configuration freeze:

1. Per-task baseline metrics with completion, refined completion, depth,
   consistency, alignment, and refined alignment.
2. Per-stage reach, first-attempt accuracy, post-repair accuracy, repair
   conversion, abstention, and damage.
3. Verifier calibration and transfer matrices with denominators and confidence
   intervals.
4. Full `B0`–`B5` compute table with logical calls, live calls, generated
   tokens, wall time, peak memory, and success per 1,000 tokens.
5. Cumulative stage-addition and gate-removal ablations.
6. Task-stratified and recognition-versus-measurement results.
7. Paired bootstrap 95% confidence intervals and stochastic-seed sensitivity.
8. Failure taxonomy with representative de-identified cases and verifier
   decisions.
9. Prompt templates, exact feedback strings, parser rules, and policy hashes.
10. Environment and dependency manifest plus commands for every table.

## H. Current artifact index

| Result | Documentation | Machine-readable artifacts |
|---|---|---|
| Golden reproduction | `PATH1_IMPLEMENTATION_AND_ABLATION_PLAN.md` | `outputs/medgemma4b_path1_reproduction` |
| Oracle diagnostics | `docs/path1_oracle_stage_probe.md`, `docs/path1_oracle_end_to_end_pilot.md` | `outputs/path1_oracle_*` |
| Stage-4 practical verifier | `docs/path1_practical_final_verifier.md` | `outputs/path1_practical_final_*` |
| Inclusion Stage 2 | `docs/path1_independent_inclusion_verifier.md` | `outputs/path1_locked_independent_inclusion_audit`, `outputs/path1_locked_independent_stage2_ablation` |
| Inclusion Stage 3 | `docs/path1_independent_stage3_verifier.md` | `outputs/path1_independent_stage3_calibration`, `outputs/path1_locked_independent_stage3_*` |
| CTR and projection Stage 3 | `docs/path1_additional_independent_stage3_verifiers.md` | `outputs/path1_independent_ctr_calibration`, `outputs/path1_independent_projection_calibration`, `outputs/path1_locked_independent_projection_audit` |
| Projection upstream gates and repair policy | `docs/path1_projection_upstream_pipeline.md` | `outputs/path1_projection_bodypart_calibration`, `outputs/path1_projection_upstream_audit`, `outputs/path1_projection_cumulative_ablation`, `outputs/path1_projection_full_cumulative_ablation`, `outputs/path1_projection_repair_policy_ablation`, `outputs/path1_projection_selected_policy`, `outputs/path1_projection_stage2_repair_policy`, `outputs/path1_projection_stage2_strict_confirmation`, `outputs/path1_projection_full_restart_control` |
| Frozen projection v1 | `docs/path1_projection_frozen_configuration.md` | `configs/path1_projection_frozen_v1.json`, `outputs/path1_projection_config_freeze/verification.json` |
| Frozen inclusion v1 | `docs/path1_inclusion_frozen_configuration.md` | `configs/path1_inclusion_frozen_v1.json`, `outputs/path1_inclusion_unified`, `outputs/path1_inclusion_full_restart_control`, `outputs/path1_inclusion_config_freeze/verification.json` |
| Frozen Carina selector v2 | `docs/path1_safe_stage2_selector_b5.md` | `configs/path1_carina_selector_frozen_v2.json`, `outputs/path1_safe_stage2_selector_b5`, `outputs/path1_carina_selector_control`, `outputs/path1_carina_selector_frozen_v2` |
| All-task support gate | `docs/path1_task_support_matrix.md` | `configs/path1_task_support_v6.json`, `outputs/path1_task_support_matrix_v6` |
| Rejected Cardiomegaly Stage 2 | `docs/path1_cardiomegaly_stage2_verifier.md` | `outputs/path1_cardiomegaly_stage2_calibration`, `outputs/path1_locked_independent_cardiomegaly_stage2_audit` |
| Rejected Mediastinal Widening Stage 2 | `docs/path1_mediastinal_stage2_verifier.md` | `outputs/path1_mediastinal_stage2_calibration`, `outputs/path1_locked_independent_mediastinal_stage2_audit` |
| Remaining Stage-2 candidates | `docs/path1_remaining_stage2_verifiers.md` | `outputs/path1_remaining_stage2_verification`, task-specific calibration/audit/ablation folders |
| Original-metric supported-task overlay | `docs/path1_official_metric_adapter.md` | `outputs/path1_official_metrics_baseline_regression`, `outputs/path1_official_metrics_inclusion_v1`, `outputs/path1_official_metrics_projection_v1`, `outputs/path1_official_metrics_supported_v1` |
| Compute controls | `docs/path1_stage2_compute_controls.md`, `docs/path1_stage2_retry_control.md` | `outputs/path1_stage2_compute_controls`, `outputs/path1_stage2_retry_control` |
| Unified Inclusion B1 | `docs/path1_inclusion_unified_b1.md` | `outputs/path1_inclusion_unified_b1` |
| Unified Inclusion B2 | `docs/path1_inclusion_unified_b2.md` | `outputs/path1_inclusion_unified_b2` |
| Projection B1/B2 controls | `docs/path1_projection_compute_controls.md` | `outputs/path1_projection_b1`, `outputs/path1_projection_b2` |
| Supported final run v2 | `docs/path1_supported_final_run_v2.md` | `configs/path1_supported_final_run_v2.json`, `outputs/path1_end_to_end_readiness_v2`, `outputs/path1_official_metrics_supported_final_v2` |
| Protocol coverage expansion | `docs/path1_protocol_coverage_expansion.md` | `outputs/path1_protocol_stage1_pilot`, `outputs/path1_inspiration_ensemble_v4`, `outputs/path1_inspiration_stage4_replay`, task-specific calibration and transfer-audit folders |
| All-task selective Stage 1 | `docs/path1_selective_protocol_stage1_all_tasks.md` | `outputs/path1_selective_protocol_stage1_all_tasks`, `outputs/path1_inspiration_measurement_prompt_ablation` |
| Selective composed overlay | `docs/path1_selective_composed_overlay.md` | `outputs/path1_selective_composed_overlay`, `outputs/path1_official_metrics_selective_composition_ablation`, `outputs/path1_official_alignment_shift_audit` |

## I. Appendix freeze checklist

- [x] Freeze the complete supported-task verifier set.
- [x] Add and audit the projection Stage-1 criterion gate.
- [x] Run the projection Stage-1/1.5 retry-budget, feedback, and blind-call pilot.
- [x] Confirm the selected per-stage projection retry policy end to end.
- [x] Run the projection Stage-2 prompt/retry and blind-call pilot.
- [x] Confirm the exploratory strict Stage-2 policy end to end.
- [x] Run and verify the projection retry-call-matched full-chain restart control.
- [x] Freeze and hash-verify the complete projection v1 configuration.
- [x] Run the 100-case cumulative inclusion experiment and matched restart control.
- [x] Freeze and hash-verify the complete inclusion v1 configuration.
- [x] Add an explicit all-task support/pass-through gate.
- [x] Freeze all prompts, thresholds, model revisions, seeds, and scorers for the supported-task overlay.
- [x] Run the applicable compute- and action-matched controls.
- [x] Run the final supported-task 1,200-case matrix once.
- [ ] Produce paired confidence intervals and per-task tables.
- [ ] Export machine-readable tables directly from verified artifacts.
- [x] Export frozen supported-task trajectories into the original metric layout.
- [x] Run and independently verify selective protocol Stage 1/1.5 on all 1,200 cases.
- [x] Run the compute-matched Inspiration Stage-3 focused-prompt/control ablation.
- [x] Compose compatible frozen downstream policies onto selective Stage 1 and verify all 1,200 cases.
- [x] Run and verify cumulative original-layout metrics and the Alignment denominator audit.
- [ ] Add representative failure figures without protected health information.
- [ ] Record exact software and hardware environment.
- [ ] Verify every appendix number against an artifact hash.

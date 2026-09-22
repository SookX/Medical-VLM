# Path 1 twelve-task completion program

## Objective and claim boundary

The engineering objective is to raise deterministic Path 1 Completion from the
current 6.25 while producing a controller that can be transferred unchanged to
additional generator backbones. Each task is handled sequentially and is
frozen before the next task is composed.

The current 6.25 result remains a development result. CXReasonBench has been
inspected repeatedly, and the Cardiomegaly increment includes retrospective
paired selection. It must not be presented as an independent confirmatory
estimate. Benchmark cases may diagnose failures and measure development
progress; model selection and confidence thresholds must use external data.

## Common workflow for every task

Each numbered task follows the same gates:

1. Audit stage reach, first failures, prompt turns, and the previous candidate.
2. Define independent visual evidence and a patient-disjoint
   calibration/validation/sealed-test split.
3. Train or calibrate without CXReasonBench inputs or outcomes.
4. Freeze model, thresholds, rules, source hashes, and routing policy.
5. Run a reference-preservation transfer audit. Any false repair disables the
   candidate; thresholds are not retuned on benchmark outcomes.
6. Run Stage 2, then cumulative Stage 2+3 and Stage 2+3+4 ablations.
7. Compare native, compute-matched retry, forced-change/no-evidence,
   always-tool, and selective-tool controls.
8. Compose only a fixed input/evidence-based policy. Never select individual
   benchmark rows because their observed outcome improves.
9. Re-run the 1,200-case deterministic metrics, paired bootstrap, compute, and
   provenance verification.

Completion is the primary endpoint. Depth, refined metrics, Consistency,
Alignment, damage, abstention, calls, tokens, latency, and memory are required
secondary endpoints. A task may be frozen with case-level abstention, but not
with an unsupported applicable stage.

## Sequential task queue

| # | Task | Current completion | First priority | Required independent evidence | Exit criterion |
|---:|---|---:|---|---|---|
| 1 | Aortic-knob Enlargement | 0 | Stage 2 | Aortic-knob x endpoints and tracheal endpoints | Safe Stage 2, width-ratio Stage 3, threshold Stage 4 |
| 2 | Ascending-aorta Enlargement | 0 | Stage 2, including second turn | Heart/trachea reference and ascending-aorta region | Both anatomy turns covered; area ratio computed |
| 3 | Cardiomegaly | 3 | Stage 3 | Frozen heart/thorax landmarks | CTR interval and view-aware final rule frozen |
| 4 | Carina Angle | 8 | Stage 3 | Three carina landmarks | Angle interval and final boundaries frozen |
| 5 | Descending-aorta Enlargement | 1 | Stage 2 | Descending-aorta and tracheal endpoints | Zero-harm selector and width ratio frozen |
| 6 | Descending-aorta Tortuosity | 1 | Stage 2 | Six serial aortic landmarks | Curvature computation and threshold frozen |
| 7 | Inclusion | 16 | Coverage expansion | Six lung-field boundary landmarks | Existing 16 preserved; added coverage passes controls |
| 8 | Inspiration | 12 | Stage 3 | Posterior ribs, diaphragm, mid-clavicular reference | Rib-count transfer gate and final rule frozen |
| 9 | Mediastinal Widening | 0 | Stage 2 | Mediastinal and level-specific thoracic endpoints | AP/PA MCR pipeline frozen |
| 10 | Projection | 26 | Robustness only | Scapula/lung masks | Existing full path revalidated without row selection |
| 11 | Rotation | 1 | Stage 2 | Medial clavicles and thoracic midline | Symmetry ratio pipeline frozen |
| 12 | Trachea Deviation | 0 | Stage 2 | Nine centerline points and thoracic midline | Signed direction aggregation frozen |

This order is stable for bookkeeping and wrap-up. Cumulative metric runs happen
after each accepted task; rejected tasks remain documented and disabled.

## Task 1: Aortic-knob Enlargement

The current candidate completes 0/100 cases and first fails Stage 2 in 99/100.
The previous generic candidate was rejected because it made one wrong
conclusive trachea mapping among 799 external validation candidates. Retuning
that candidate on the observed error would be leakage.

The replacement is an explicit landmark model rather than an overlay-class
nearest-neighbour mapper. The local external inventory contains 535 NIH images
with exact aortic-knob x endpoints and tracheal endpoint annotations. Script
`scripts/prepare_path1_aortic_knob_landmarks_v2.py` creates a patient-disjoint
65/17/18 calibration/validation/sealed-test manifest and records every source
hash. CXReasonBench is excluded.

Task 1 execution steps:

1. Prepare and verify the sealed landmark cohort.
2. Train a four-output landmark regressor or heatmap decoder with calibration
   data only; select epoch and uncertainty calibration on validation only.
3. Unseal the external test once. Report endpoint error, knob/trachea width
   error, ratio MAE, interval accuracy, and AP/PA strata.
4. Render predicted spans in the exact benchmark option representation and
   freeze a confidence-abstaining Stage-2 selector.
5. Audit all correct benchmark reference candidates for zero false repairs.
6. Run fixed-policy Stage-2, Stage-2+3, and Stage-2+3+4 ablations plus controls.
7. If accepted, compose the whole fixed task policy into candidate v5 and run
   official metrics. If rejected, retain the negative result and move to Task 2.

### Task 1 progress (2026-09-20)

- Steps 1–3 are complete.
- The prepared cohort contains 535 patient-disjoint NIH images: 362
  calibration, 84 validation, and 89 sealed test.
- The validation-selected checkpoint is epoch 14. Validation normalized
  coordinate MAE is 0.012762 and aortic-knob/trachea ratio MAE is 0.171757.
- The checkpoint was then evaluated once on the 89-image external test:
  normalized coordinate MAE 0.012662 and ratio MAE 0.168934 (AP 0.166166; PA
  0.170815). Test data did not influence model selection.
- Source and result artifacts are under
  `data/path1_aortic_knob_landmarks_v2`,
  `outputs/path1_aortic_knob_landmarks_v2_training`, and
  `outputs/path1_aortic_knob_landmarks_v2_test`.
- Steps 4–7 remain: benchmark-format candidate matching, uncertainty
  calibration, zero-false-repair transfer audit, cumulative ablations, and
  fixed-policy composition. The external result alone does not establish a
  Completion gain.
- The benchmark-format audit found 100 first anatomy turns with four displayed
  candidates each; 66 cases request a second four-candidate anatomy turn. The
  Stage-2 selector must therefore retain dialogue state and cover both turns,
  rather than reproduce the earlier first-turn-only mistake.

Task 1 was subsequently rejected at the locked transfer gate. The frozen
selector produced only 1 correct conclusive path and 99 wrong conclusive paths
on the 100 benchmark-format cases. This is a severe external-to-benchmark
candidate-rendering mismatch despite good external landmark error. The model,
matcher, and thresholds remain archived but disabled; they were not retuned on
benchmark answers. Task 1 therefore remains 0/100 and the program proceeds to
Task 2.

## Task 2 progress (2026-09-21)

- Reusing the pre-existing zero-false-repair ensemble as a stateful selector
  was safe but ineffective: 0 correct conclusive, 0 wrong, 100 abstentions.
- A new external heart/trachea landmark cohort was prepared with 138
  calibration, 21 validation, and 35 sealed-test images.
- The validation-selected epoch-12 landmark model achieved normalized point
  distance 0.019432 on validation and 0.017528 on the once-unsealed test.
- A representation-faithful selector replaced the old vertical-line proxy
  with the actual annotated heart-to-trachea reference line. It passed its
  external gate: 76/79 correct conclusive with zero errors on calibration and
  12/12 with zero errors on validation.
- Locked benchmark-format transfer was a near miss: 98 correct conclusive, 1
  wrong conclusive, and 1 abstention. Under the predeclared zero-false-repair
  gate it is rejected and remains disabled.
- Requiring agreement from the older frozen ensemble removed the error but
  also removed all coverage (100/100 abstentions), so that consensus policy is
  rejected as ineffective.
- No confidence threshold or special-case rule was fitted to the single
  benchmark error. Task 2 remains 0/100 and work proceeds to Task 3.

## Task 3 progress (2026-09-21)

- The previously trained CTR landmark model was evaluated through the existing
  independent NIH calibration protocol. It reduced validation false repairs
  from 2/71 to 1/71 but retained a very broad mean plausible interval of
  0.353475 and detected only 3.87% of wrong alternatives. It remains rejected.
- A pre-specified AP/PA-specific affine calibration narrowed the mean
  validation interval to 0.211748, but produced 3/71 false repairs, all in the
  AP stratum. It is rejected.
- A PA-only rule was not promoted because choosing that subgroup after viewing
  validation outcomes would require a new untouched confirmation cohort.
- Cardiomegaly therefore keeps its accepted Stage-2 contribution, while Stage
  3 remains rejected and Stage 4 remains inactive. Work proceeds to Task 4.

## Task 4 progress (2026-09-21)

- A patient-disjoint external cohort was prepared with 203 calibration, 58
  validation, and 69 sealed-test radiographs carrying exact three-point carina
  annotations.
- The epoch-9 checkpoint achieved angle MAE 9.909853 degrees on validation and
  10.726919 degrees on the once-unsealed external test. Coordinate MAE was
  0.006170 on test.
- A locked affine angle correction and residual-extrema interval preserved all
  58 validation and all 69 external-test reference ranges with zero false
  repairs.
- The safe interval was 61.979347 degrees wide and uniquely selected 0/58
  validation and 0/69 test options. It therefore failed the predeclared
  positive-efficacy gate and was rejected before benchmark transfer.
- Carina retains its accepted Stage-2 selector and 8/100 completion. Stage 3
  is now explicitly `rejected`, not `unsupported`; Stage 4 remains inactive.
- Because Tasks 1–4 yielded no newly accepted end-to-end intervention, no
  candidate v5 is composed. Candidate v4 remains the last valid 1,200-case
  candidate.

## Transfer to additional models

## Five-pair execution outcome (2026-09-21)

The five planned two-task runs have all been executed. Each is currently
`partial_gate_blocked`, because one member or one downstream measurement stage
failed its predeclared gate. Accepted components were never promoted across a
failed measurement boundary.

| Pair | Accepted evidence | Rejected boundary |
|---|---|---|
| Cardiomegaly + Carina | Cardiomegaly Stage 3/4, +3 internal completions | Carina angle failed confirmation |
| Inspiration + Aortic Knob | Inspiration Stage 3/4, +1; Aortic Stage 2 at 48/100 | Aortic measurement probe 0/14 |
| Ascending + Descending Aorta | Ascending full path 98/100; Descending Stage 2 at 65/100 | Descending displayed-width ratio 7/38 |
| Mediastinal + Rotation | Mediastinal Stage 2 at 20/100; Rotation full path at 4 eligible cases, raising completion 1 to 5 | Mediastinal Stage 3 ratio remains unsafe; Rotation requires untouched confirmation |
| Tortuosity + Tracheal Deviation | Tortuosity Stage 2 at 52/100; Tracheal Deviation Stage 2 hybrid at 2/100, both zero-error | curvature unrecoverable from region mask; external direction model had no positive zero-error OOF policy |

The composed diagnostic overlay reaches Completion 14.31, but the all-task
release gate remains closed. The result and its explicit claim boundary are
verified by `scripts/verify_path1_final_candidate_v5_diagnostic.py`.

Only after the twelve task policies and routing thresholds are frozen should
the generator backbone change. Each new model receives identical cases,
prompts, evidence, direct selectors, abstention thresholds, scorer, and case
order. Report native and controlled pipeline results separately for every
backbone. No model-specific threshold tuning is allowed on CXReasonBench.

## Three Stage-3 geometry rescues (2026-09-22)

The requested Aortic-knob ratio, Carina angle, and Descending-aorta ratio
paths are implemented in `cxreason/path1/measurement_geometry.py`. The ratio
operator follows the benchmark criterion exactly: maximum target width divided
by median tracheal width. The Carina operator skeletonizes the displayed mask
and measures its two inferior rays.

Standalone transfer is imperfect and is not promoted as a zero-error
verifier: Aortic is 33 correct / 12 wrong / 3 abstain among 48 Stage-2-eligible
cases; Carina is 32 / 48 / 20 across 100; Descending Aorta is 55 / 9 / 1 among
65. The accepted engineering policy therefore runs only as a rescue after a
native trajectory has already failed. It never replaces a completed case.
End-to-end development ablations improve completion without binary completion
losses: Aortic 0 to 24, Carina 8 to 11, and Descending Aorta 1 to 48.

The composed v7 diagnostic scores Completion 20.13 and refined Completion
13.88 over 1,200 cases. This is not a release result: refined Completion is
below v6 (14.14), standalone measurement errors remain, and the benchmark was
used for development. Use v7 as evidence that the measurements are useful,
then add external/untouched confidence calibration before freezing them for a
paper claim.

### Frozen consensus revision

The final selective revision uses four fixed red-overlay thresholds, retains
only the largest connected component, and requires every surviving geometry
estimate to select the same measurement option. A second interval-containment
gate requires the complete option interval to imply one Stage-4 diagnosis.

- Aortic Knob: 5 gains, 0 losses; Completion 0 to 5.
- Descending Aorta: 22 gains, 0 losses; Completion 1 to 23.
- Carina: two zero-error Stage-3 consensus cases exist, but neither intersects
  an unfinished measurement trajectory. The finalized policy abstains and
  preserves its existing Completion of 8.

The composed v9 diagnostic has Completion 16.75, refined Completion 14.00,
Consistency 41.25, Alignment 42.13, and refined Alignment 24.83. Compared with
raw v7, it deliberately exchanges coverage for reliability. It remains a
development result requiring untouched confirmation.

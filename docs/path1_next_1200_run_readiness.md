# Path 1 next 1,200-case run readiness

## Five-pair diagnostic update (2026-09-21)

The subsequent Rotation repair produces diagnostic v6: Completion **14.61**,
Refined Completion **14.14**, Depth **1.53**, Consistency **52.56**, Alignment
**31.11**. Rotation contributes four additional completions with zero losses.
Its disconnected-clavicle repair was developed after inspecting a transfer
failure, so the untouched-confirmation caveat remains. The verified artifact
is `outputs/path1_final_candidate_v6_diagnostic_official`.

The later five-pair program produced a new **diagnostic** 1,200-case overlay.
It is verified at 100 cases for each of all 12 tasks and uses only three newly
accepted zero-regression end-to-end replacements: Cardiomegaly, Inspiration,
and Ascending-aorta Enlargement. Under the repository's deterministic official
metric adapter it scores:

- Completion 14.31 (v4: 6.25; local native baseline: 1.73)
- Refined Completion 13.82
- Depth 1.52 and Refined Depth 1.51
- Consistency 48.92
- Alignment 31.11 and Refined Alignment 27.21

Artifacts are in `outputs/path1_final_candidate_v5_diagnostic_official`.
This result exceeds the earlier 7--8 Completion development target, but it is
not the all-task release: only the promoted tasks change outcomes, six
task-specific measurement cells remain rejected, and Ascending-aorta rules
were revised after inspecting a benchmark transfer error. It is therefore a
development upper-bound/diagnostic result, not an untouched confirmatory
estimate and not directly comparable to the retired Gemini-scored paper row.

The full suite now passes 245 tests and 2,441 subtests. Reproduce verification
with `python scripts/verify_path1_final_candidate_v5_diagnostic.py` followed by
the pytest command below.

## Decision

The repository is finalized for the next credible 1,200-case experiment at
the last accepted policy boundary: `outputs/path1_final_candidate_v4`.

No candidate v5 is created. The four sequential improvement attempts did not
pass all external safety, benchmark-format transfer, and positive-efficacy
gates. Re-running 1,200 cases under an unchanged policy would duplicate v4 and
is not a new experiment.

## Frozen result

Candidate v4 has deterministic official metrics:

- Completion: 6.25
- Refined Completion: 5.76
- Depth: 1.26
- Refined Depth: 1.26
- Consistency: 47.85
- Alignment: 26.66
- Refined Alignment: 22.27

This is a development result, not an independent confirmatory estimate. The
benchmark has been inspected repeatedly, and the Cardiomegaly increment uses
retrospective paired row selection. The next scientifically meaningful full
experiment is either a fixed-policy transfer to another generator backbone or
evaluation on untouched cases, not another identical MedGemma replay.

## Release audit (2026-09-21)

The current package passed all release checks:

- Candidate verification: `verified`, 1,200 cases, 12 tasks, canonical artifact
  index SHA-256
  `3bd6cc0556ead2cafa70f03a99599a7748780370b2def320de4f2e203dfc718b`.
- Official-metric increment verification: `verified`, two arms, summary SHA-256
  `05b1ffe4c94a89eb1c73a8b8ffc63d5e28390b5b64acd5ef3c2a34c82143e76f`.
- Task-by-stage contract: `verified_preflight_contract`, 12 tasks, 1,200 cases,
  and all 60 task-stage cells checked. Only Inclusion and Projection satisfy
  the genuine-full-path gate.
- Test suite: 240 tests and 2,441 subtests passed.

Reproduce the release audit with:

```powershell
python scripts\verify_path1_final_candidate_v4.py
python scripts\verify_path1_final_candidate_v4_official_increment.py
python scripts\audit_path1_task_stage_program_v1.py
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q
```

`verify_path1_end_to_end_readiness_v2.py` remains valid for the earlier
three-task v2 overlay, but it is a legacy verifier and must not be cited as the
v4 release audit.

## Finalized sequential attempts

| Task | Result | Deployment |
|---|---|---|
| Aortic-knob | 1 correct / 99 wrong transfer paths | Rejected |
| Ascending-aorta | 98 correct / 1 wrong / 1 abstain | Rejected by zero-error gate |
| Cardiomegaly Stage 3 | 1/71 false repair; view-aware 3/71 | Rejected |
| Carina Stage 3 | 0 false repairs but 0/69 conclusive | Rejected for no efficacy |

All rejected components remain disabled. No benchmark-specific threshold or
case exception was added to rescue them.

## What can now be run credibly

1. Reproduce and verify candidate v4 exactly from its frozen artifacts.
2. Apply the same frozen controller to a different model backbone, reporting
   native and controlled results separately.
3. Evaluate the frozen policy on a genuinely untouched end-to-end cohort.

For cross-model transfer, task policies, confidence thresholds, direct
selectors, abstention behavior, prompts, scorer, case order, and parsing must
remain unchanged. Model-specific tuning on CXReasonBench is prohibited.

## What is not claimed

- Candidate v4 is not an exact reproduction of the paper's retired Gemini
  reasoning scorer.
- The deterministic local scorer is the supported evaluation used here.
- Only Inclusion and Projection currently have fully frozen applicable paths.
- The project is ready for the next full controlled experiment, but it is not
  yet an all-12-task fully supported release.

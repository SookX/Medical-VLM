# Path 1 Task-Support Matrix

The task-support gate prevents a verifier or repair policy validated for one
task from being silently reused on another. Its source configuration is
`configs/path1_task_support_v6.json`; the expanded machine-readable audit and
rendered table are under `outputs/path1_task_support_matrix_v6`.

## Current coverage

| Stage | Repair-enabled tasks | Frozen | Validated, not frozen | Rejected | Unsupported |
|---|---:|---:|---:|---:|---:|
| Stage 1 | 1/12 | 1 | 0 | 0 | 11 |
| Stage 1.5 | 1/12 | 1 | 0 | 0 | 11 |
| Stage 2 | 3/12 | 3 | 0 | 9 | 0 |
| Stage 3 | 2/12 | 2 | 0 | 1 | 9 |
| Stage 4 | 11/12 | 2 | 9 | 0 | 1 |

Repair is enabled in 18/60 task-stage cells. Projection, Inclusion, and Carina
Angle are now frozen. Inclusion's unified 100-case policy raises Completion from 10%
to 16% and mean Depth from 0.85 to 1.01 with no observed depth regression.
Carina's independent selector raises Completion from 6% to 8% and mean Depth
from 0.82 to 0.92 with six depth wins and no losses. The other nine tasks are
Stage-4-only engineering configurations. Their Stage-2 candidates are now
explicitly rejected rather than unsupported. The eight
newly closed candidates comprise two external-validation failures, three
locked-transfer failures, and three safe but ineffective repair policies.
Cardiomegaly Stage 3 is also rejected.

Unsupported and rejected stages follow a mandatory pass-through policy: use
the native model answer, instantiate no verifier, and request no retry. The
support-policy API raises an error if a runner requests repair at one of these
stages.

## Frozen-baseline reach

| Stage group | Cases/attempts | Correct | Incorrect | Unscored |
|---|---:|---:|---:|---:|
| Initial diagnosis | 1,200 | 555 | 645 | 0 |
| Stage 1 criterion | 1,200 | 477 | 723 | 0 |
| Stage 1.5 applicability | 269 | 129 | 0 | 140 |
| Stage 2 anatomy | 337 | 36 | 299 | 2 |
| Stage 3 measurement | 36 | 19 | 17 | 0 |
| Stage 4 final | 19 | 17 | 2 | 0 |

Stage 2 is the main visual bottleneck among tasks without validated grounding:

| Priority | Task | Baseline Stage-2 reach | Incorrect or unscored |
|---:|---|---:|---:|
| 1 | Cardiomegaly | 50 | 50 |
| 2 | Mediastinal widening | 49 | 49 |
| 3 | Ascending-aorta enlargement | 38 | 38 |
| 4 | Rotation | 33 | 31 |
| 5 | Trachea deviation | 25 | 25 |

This ranking measures observed opportunity, not verifier feasibility. A task
must still pass independent calibration, locked transfer, repair, damage, and
compute-control gates before support is enabled.

## Run gate

A diagnostic 1,200-case run is allowed with unsupported stages passing
through. The frozen Carina/Inclusion/Projection supported-task overlay is
complete under `configs/path1_supported_final_run_v2.json`, with all other tasks passing
through natively. A final all-task intervention remains blocked because:

- nine tasks have rejected Stage-2 repair policies and only Stage-4 repair support;
- the paper-versus-local baseline metric difference remains unresolved.

Carina Angle, Inclusion, and Projection are currently in `final_ready_tasks`;
their applicable controls are complete and permanently verified. Stage-4-only support
on the other nine tasks is still an engineering component rather than a frozen
task configuration.

## Reproduction and verification

```powershell
python scripts\audit_path1_task_support.py --config configs\path1_task_support_v6.json --output-dir outputs\path1_task_support_matrix_v6
python scripts\verify_path1_task_support_v6.py
```

The verifier recomputes the matrix from all 1,200 frozen scoring artifacts,
checks all evidence paths, validates task and stage inventories, and asserts
that the unsafe final-run gate remains closed.

Experiment runners should load the policy and validate requested stages before
constructing an executor:

```python
from pathlib import Path
from cxreason.path1.support_policy import load_task_support_policy

policy = load_task_support_policy(Path("configs/path1_task_support_v6.json"))
repair_stages = policy.validate_requested_stages(
    "inclusion", ["stage2", "stage3", "stage4"]
)
```

Cardiomegaly and Mediastinal Widening failed locked transfer with 16/166 and
3/166 false repairs. Of the remaining eight candidates, Aortic-Knob Enlargement
and Trachea Deviation failed external validation; Rotation, Descending-Aorta
Enlargement, and Descending-Aorta Tortuosity failed locked transfer; and
Ascending-Aorta Enlargement, Carina Angle, and Inspiration safely routed 20
repairs but corrected none. Details and hashes are in
`docs/path1_remaining_stage2_verifiers.md` and
`outputs/path1_remaining_stage2_verification/summary.json`.

The later B5 mechanism experiment did not retune those policies. It directly
used the complete option set from each safe selector. This improved only Carina
Angle; the result passed a forced-change no-evidence control and was promoted
as the single new Stage-2 cell in support v6. Ascending Aorta and Inspiration
remain rejected because their downstream anatomy turns are uncovered.

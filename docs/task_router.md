# Deterministic Task Router

The project uses a deterministic task router rather than a learned MoE.

```text
task name -> TaskSpec -> stage-specific gates
```

The controller remains generic. It only knows how to run:

```text
generate -> verify -> PASS/REPAIR -> retry current stage or continue
```

Task-specific medical knowledge lives in `cxreason/tasks/registry.py`.
The dependency audit in `cxreason/controller/dependency_audit.py` can replay a
completed accepted chain against the same task spec after local repair.

## Registered Tasks

Each task spec defines:

- preferred Stage 1 criterion
- accepted criterion terms
- required Stage 2 anatomy fields
- required Stage 3 measurement/recognition fields
- optional arithmetic consistency checks
- final decision field

The current 12 task experts are:

| Task | Stage 2 Anatomy Schema |
| --- | --- |
| `cardiomegaly` | `heart_xmin`, `heart_xmax`, `lung_xmin`, `lung_xmax`, `viewposition` |
| `mediastinal_widening` | `mediastinum_xmin`, `mediastinum_xmax`, `lung_xmin`, `lung_xmax`, `viewposition` |
| `carina_angle` | `point_1`, `point_2`, `point_3`, `viewposition` |
| `trachea_deviation` | `point_1` ... `point_9`, `viewposition` |
| `aortic_knob_enlargement` | `aortic_knob_xmin`, `aortic_knob_xmax`, `trachea_point_right`, `trachea_point_left`, `viewposition` |
| `ascending_aorta_enlargement` | `heart_point`, `trachea_point`, `viewposition` |
| `descending_aorta_enlargement` | `trachea_point_right`, `trachea_point_left`, `desc_aorta_point_right`, `desc_aorta_point_left`, `viewposition` |
| `descending_aorta_tortuous` | `point_1` ... `point_6`, `viewposition` |
| `inclusion` | lung apex, side, and bottom landmark points for both lungs plus `viewposition` |
| `inspiration` | `mid-clavicular_line(x)`, `viewposition` |
| `rotation` | clavicle medial endpoints, spinous/midline points, same-level midline points, `viewposition` |
| `projection` | right/left scapular regions and right/left scapula-lung overlap regions plus `viewposition` |

## Verification Levels

The current registry supports several verifier strengths:

- `structural`: required fields are present and typed correctly.
- `sanity`: structural checks plus broad value-domain checks.
- `arithmetic`: reported ratios match component measurements.
- `clinical-rule`: deterministic final rule derived from accepted Stage 3 evidence.
- `reference/oracle`: reserved for official CXReasonBench answers later.

Each `TaskSpec` exposes `verification_levels`, so corruption metrics can be
reported separately for structural-only checks and stronger semantic checks.

All 12 registered tasks currently have Stage 4 clinical-rule gates. Stage 3 is
either arithmetic or sanity-checked depending on whether the measurement can be
derived from component fields.

CheXStruct-derived values are benchmark-linked development evidence. They should
not be described as independent visual verification.

## Useful Commands

Run any task through the generic mock router:

```powershell
python scripts\run_mock_task_pipeline.py --task cardiomegaly --row-index 1
python scripts\run_mock_task_pipeline.py --task rotation --row-index 0
python scripts\run_mock_task_pipeline.py --task mediastinal_widening --row-index 0
```

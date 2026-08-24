# Deterministic Clinical-Rule Gates

The current mock verifier derives these Stage 4 rules from CheXStruct label
boundaries. They are development scaffolding until official CXReasonBench
criteria are plugged in.

| Task | Final Rule |
| --- | --- |
| `cardiomegaly` | PA: `ctr >= 0.495`; AP: `ctr >= 0.545` |
| `mediastinal_widening` | PA: `mcr >= 0.245`; AP: `mcr >= 0.325` |
| `aortic_knob_enlargement` | `ratio > 2.495` |
| `ascending_aorta_enlargement` | `ratio >= 0.1` |
| `descending_aorta_enlargement` | `ratio >= 2.5` |
| `descending_aorta_tortuous` | `curvature >= 0.00085` |
| `carina_angle` | abnormal if `angle < 39.5` or `angle > 80.5` |
| `inspiration` | inadequate if `rib_position <= 8` |
| `rotation` | rotation if `ratio < 0.395` |
| `projection` | large scapular overlap if `max(ratio_right, ratio_left) >= 0.3` |
| `inclusion` | sufficient only if all six regional inclusion labels are `1` |
| `trachea_deviation` | deviation if `direction != "flat"` |

The boundaries above matched CheXStruct labels with zero mismatches over the
public rows checked locally.

CheXStruct is benchmark-linked evidence, not independent visual verification.

## Stage 3 Sanity Checks

Tasks without arithmetic Stage 3 checks now use broad value-domain sanity gates:

| Task | Stage 3 Sanity Check |
| --- | --- |
| `ascending_aorta_enlargement` | `ratio` must be in `[0, 1]` |
| `carina_angle` | `angle` must be in `[0, 180]` degrees |
| `descending_aorta_tortuous` | `curvature` must be in `[0, 0.02]` |
| `inclusion` | six regional labels must be binary; six ratios must be in `[0, 1]` |
| `inspiration` | `rib_position` must be in `[0, 12]` |
| `trachea_deviation` | `direction` must be one of `flat`, `left`, `right`, `left&right`, `right&left` |

These checks catch impossible structured measurements but do not prove visual
grounding.

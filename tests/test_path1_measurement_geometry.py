import numpy as np

from cxreason.path1.measurement_geometry import carina_angle_extrema, carina_angle_from_mask, carina_angle_pca, largest_component, maximum_to_median_width_ratio


def test_maximum_to_median_width_ratio() -> None:
    target = np.zeros((5, 10), dtype=bool); target[1, 2:7] = True; target[2, 3:6] = True
    trachea = np.zeros((5, 10), dtype=bool); trachea[1, 4:6] = True; trachea[2, 4:7] = True; trachea[3, 4:6] = True
    assert maximum_to_median_width_ratio(target, trachea) == 2.5


def test_carina_angle_from_thin_right_angle_mask() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    for offset in range(5):
        mask[2 + offset, 4 - offset] = True
        mask[2 + offset, 4 + offset] = True
    assert abs(carina_angle_from_mask(mask) - 90.0) < 1e-6
    assert abs(carina_angle_extrema(mask) - 90.0) < 1e-6
    assert abs(carina_angle_pca(mask) - 90.0) < 1e-6


def test_largest_component_removes_overlay_noise() -> None:
    mask=np.zeros((8,8),dtype=bool); mask[2:5,2:5]=True; mask[7,7]=True
    cleaned=largest_component(mask)
    assert cleaned.sum()==9 and not cleaned[7,7]

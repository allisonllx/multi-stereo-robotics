import numpy as np
import pytest

from processing.stereo_depth import disparity_to_depth, evaluate_depth


def test_disparity_to_depth_uses_focal_and_baseline_and_masks_invalid():
    disparity = np.array([[10.0, 0.0, -1.0]], dtype=np.float32)
    depth = disparity_to_depth(disparity, focal_px=500, baseline_m=0.2)
    assert depth[0, 0] == pytest.approx(10.0)
    assert np.isnan(depth[0, 1]) and np.isnan(depth[0, 2])


def test_evaluate_depth_reports_metric_errors_only_on_valid_pixels():
    estimate = np.array([[1.1, 2.0], [3.0, np.nan]])
    reference = np.array([[1.0, 2.5], [0.0, 4.0]])
    metrics = evaluate_depth(estimate, reference)
    assert metrics["valid_pixels"] == 2
    assert metrics["mae_m"] == pytest.approx(0.3)
    assert metrics["rmse_m"] == pytest.approx(np.sqrt((0.1**2 + 0.5**2) / 2))

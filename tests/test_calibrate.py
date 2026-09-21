"""Intrinsic / stereo calibration tests on synthetic correspondences.

A production bug that would fail these tests: leaving CALIB_FIX_INTRINSIC
unset, pairing the wrong corner IDs, or reporting residuals in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.calibrate_intrinsics import calibrate_intrinsics
from calibration.calibrate_stereo import calibrate_stereo, intersect_view
from calibration.common import charuco_board_from_config
from calibration.verify_calibration import (
    adjacent_charuco_pairs,
    epipolar_distances,
    mean_vertical_disparity,
)


def _synthetic_board_views(n_views: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    config = {
        "charuco": {
            "squares_x": 5,
            "squares_y": 7,
            "square_length_m": 0.04,
            "marker_length_m": 0.03,
            "dictionary": "DICT_5X5_250",
        }
    }
    board = charuco_board_from_config(config)
    object_points = board.getChessboardCorners().astype(np.float32)
    image_size = (1280, 720)
    fx, fy, cx, cy = 900.0, 910.0, 640.0, 360.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    import cv2

    obj_views = []
    img_views = []
    for _ in range(n_views):
        rvec = rng.normal(0.0, 0.25, size=3).astype(np.float64)
        tvec = np.array(
            [rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(0.35, 0.7)],
            dtype=np.float64,
        )
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
        projected = projected.reshape(-1, 2).astype(np.float32)
        projected += rng.normal(0.0, 0.05, size=projected.shape).astype(np.float32)
        obj_views.append(object_points.copy())
        img_views.append(projected)
    return obj_views, img_views, image_size, K, dist, object_points, config


def test_intrinsics_recover_focal_length_from_synthetic_views():
    obj_views, img_views, image_size, K_true, dist, _, _ = _synthetic_board_views()
    result = calibrate_intrinsics(obj_views, img_views, image_size)

    assert result.reprojection_error_px < 0.2
    assert result.camera_matrix[0, 0] == pytest.approx(K_true[0, 0], rel=0.05)
    assert result.camera_matrix[1, 1] == pytest.approx(K_true[1, 1], rel=0.05)
    assert result.camera_matrix[0, 2] == pytest.approx(K_true[0, 2], abs=15.0)


def test_stereo_recovers_baseline_with_intrinsics_held_fixed():
    import cv2

    obj_views, img_a, image_size, K, dist, object_points, _ = _synthetic_board_views(n_views=10, seed=1)
    R_true, _ = cv2.Rodrigues(np.array([0.02, -0.01, 0.03], dtype=np.float64))
    T_true = np.array([[0.12], [-0.004], [0.001]], dtype=np.float64)

    img_b = []
    rng = np.random.default_rng(2)
    for obj, pts_a in zip(obj_views, img_a):
        # Reconstruct a plausible rvec/tvec from camera A projections, then apply stereo.
        ok, rvec, tvec = cv2.solvePnP(obj, pts_a.reshape(-1, 1, 2), K, dist)
        assert ok
        R_a, _ = cv2.Rodrigues(rvec)
        R_b = R_true @ R_a
        t_b = R_true @ tvec + T_true
        rvec_b, _ = cv2.Rodrigues(R_b)
        projected, _ = cv2.projectPoints(obj, rvec_b, t_b, K, dist)
        projected = projected.reshape(-1, 2).astype(np.float32)
        projected += rng.normal(0.0, 0.04, size=projected.shape).astype(np.float32)
        img_b.append(projected)

    stereo = calibrate_stereo(
        obj_views,
        img_a,
        img_b,
        K,
        dist,
        K,
        dist,
        image_size,
    )

    baseline = float(np.linalg.norm(stereo.T))
    assert stereo.rms_error < 0.3
    assert baseline == pytest.approx(0.12, rel=0.08)
    np.testing.assert_allclose(stereo.camera_matrix_a, K, atol=1e-8)
    np.testing.assert_allclose(stereo.camera_matrix_b, K, atol=1e-8)


def test_intersect_view_requires_shared_ids():
    object_points = np.array([[0, 0, 0], [0.04, 0, 0], [0.08, 0, 0]], dtype=np.float32)
    ids_a = np.array([0, 1, 2])
    ids_b = np.array([2, 1])
    pts_a = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
    pts_b = np.array([[30, 3], [20, 2]], dtype=np.float32)

    obj, ima, imb = intersect_view(object_points, ids_a, pts_a, ids_b, pts_b)
    assert obj.shape[0] == 2
    assert ima[0].tolist() == pytest.approx([2.0, 2.0])
    assert imb[0].tolist() == pytest.approx([20.0, 2.0])


def test_epipolar_and_vertical_metrics_on_rectified_identity():
    pts_a = np.array([[10.0, 20.0], [40.0, 80.0]], dtype=np.float64)
    pts_b = np.array([[12.0, 20.4], [44.0, 80.2]], dtype=np.float64)
    F = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    distances = epipolar_distances(pts_a, pts_b, F)
    assert distances.shape == (2,)
    assert mean_vertical_disparity(pts_a, pts_b) == pytest.approx(0.3)


def test_adjacent_charuco_pairs_counts_grid_edges():
    # 5x7 squares -> 4x6 inner corners: 18 horizontal + 20 vertical neighbours.
    assert len(adjacent_charuco_pairs(5, 7)) == 38

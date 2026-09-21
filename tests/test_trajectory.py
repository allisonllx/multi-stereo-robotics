import numpy as np
import pytest

from processing.trajectory import camera_b_world_pose, fit_planar_rigid_alignment, geodetic_to_enu


def test_camera_b_pose_uses_inverse_of_a_to_b_extrinsic():
    world_a = np.eye(4)
    a_to_b = np.eye(4); a_to_b[0, 3] = 0.2
    world_b = camera_b_world_pose(world_a, a_to_b)
    assert world_b[0, 3] == pytest.approx(-0.2)


def test_geodetic_to_enu_has_expected_cardinal_directions():
    lat0, lon0 = 1.3, 103.8
    enu = geodetic_to_enu(np.array([lat0, lat0 + 0.00001]), np.array([lon0, lon0]), np.array([10, 10]), lat0, lon0, 10)
    np.testing.assert_allclose(enu[0], [0, 0, 0], atol=1e-6)
    assert enu[1, 1] > 1.0
    assert abs(enu[1, 0]) < 0.01


def test_fit_planar_rigid_alignment_recovers_rotation_and_translation():
    local = np.array([[0, 0], [1, 0], [1, 2], [-1, 1]], dtype=float)
    R = np.array([[0, -1], [1, 0]], dtype=float); t = np.array([4, 8])
    target = (R @ local.T).T + t
    fit = fit_planar_rigid_alignment(local, target)
    np.testing.assert_allclose(fit.rotation, R, atol=1e-10)
    np.testing.assert_allclose(fit.translation, t, atol=1e-10)
    assert fit.rmse_m < 1e-10

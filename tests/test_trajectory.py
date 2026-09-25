import numpy as np
import pytest

from processing.trajectory import align_phone_worlds, camera_b_world_pose, fit_planar_rigid_alignment, geodetic_to_enu, opencv_extrinsic_to_arkit, world_from_rectified_camera


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


def test_align_phone_worlds_recovers_constant_world_transform():
    a_to_b = np.eye(4); a_to_b[0, 3] = 0.2
    world_a = []
    for x in [0.0, 1.0, 2.0]:
        pose = np.eye(4); pose[2, 3] = x; world_a.append(pose)
    b_world_to_a_world = np.eye(4); b_world_to_a_world[:3, 3] = [4, 0, -3]
    world_b = [np.linalg.inv(b_world_to_a_world) @ camera_b_world_pose(pose, a_to_b) for pose in world_a]
    result = align_phone_worlds(world_a, world_b, a_to_b)
    np.testing.assert_allclose(result.b_world_to_a_world, b_world_to_a_world, atol=1e-10)
    assert result.translation_rmse_m < 1e-10


def test_opencv_extrinsic_is_conjugated_into_arkit_camera_axes():
    transform = np.eye(4); transform[:3, 3] = [1, 2, 3]
    converted = opencv_extrinsic_to_arkit(transform)
    np.testing.assert_allclose(converted[:3, 3], [1, -2, -3])


def test_rectified_camera_pose_includes_rectification_rotation():
    world_from_camera = np.eye(4)
    rectification = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    result = world_from_rectified_camera(world_from_camera, rectification)
    axes = np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(result[:3, :3], axes @ rectification.T @ axes)

import numpy as np

from processing.mapping import backproject_depth, voxel_downsample


def test_backproject_depth_uses_intrinsics_and_camera_to_world():
    depth = np.array([[2.0]], dtype=np.float32)
    K = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
    pose = np.eye(4); pose[:3, 3] = [1, 2, 3]
    points, pixels = backproject_depth(depth, K, pose)
    # Depth is OpenCV (+z forward); ARKit camera coordinates use -z forward.
    np.testing.assert_allclose(points, [[1, 2, 1]])
    np.testing.assert_array_equal(pixels, [[0, 0]])


def test_voxel_downsample_averages_points_and_colours():
    points = np.array([[0.01, 0, 0], [0.02, 0, 0], [1, 0, 0]])
    colours = np.array([[10, 20, 30], [30, 40, 50], [100, 110, 120]], dtype=np.uint8)
    p, c = voxel_downsample(points, colours, voxel_size_m=0.1)
    np.testing.assert_allclose(p[0], [0.015, 0, 0])
    np.testing.assert_array_equal(c[0], [20, 30, 40])

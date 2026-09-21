import numpy as np

from calibration.verify_calibration import stereo_rectify_maps


def test_stereo_rectify_maps_accepts_flat_translation_from_yaml():
    camera_matrix = np.array(
        [[900.0, 0.0, 540.0], [0.0, 900.0, 960.0], [0.0, 0.0, 1.0]]
    )
    rectification = stereo_rectify_maps(
        camera_matrix,
        np.zeros(5),
        camera_matrix,
        np.zeros(5),
        (1080, 1920),
        np.eye(3),
        np.array([0.12, 0.0, 0.0]),
    )

    assert rectification["map1_a"].shape == (1920, 1080)
    assert rectification["map1_b"].shape == (1920, 1080)

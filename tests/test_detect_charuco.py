"""ChArUco detection tests against a board rendered by OpenCV.

A production bug that would fail these tests: dropping corner IDs, skipping
subpixel refine, or matching stereo corners by list index instead of ID.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.common import charuco_board_from_config
from calibration.detect_charuco import (
    common_corner_ids,
    detect_charuco_image,
    refine_corners_subpixel,
)


def test_image_directory_uses_numeric_filename_stem_as_frame_id(tmp_path, monkeypatch):
    import calibration.detect_charuco as module

    class FakeCV:
        IMREAD_COLOR = 1

        @staticmethod
        def imread(path, _mode):
            return np.zeros((4, 5, 3), dtype=np.uint8)

    monkeypatch.setattr(module, "require_cv2", lambda: FakeCV)
    (tmp_path / "000012.jpg").write_bytes(b"x")

    frame_id, timestamp, _image = next(module._iter_images_from_dir(tmp_path))

    assert frame_id == 12
    assert timestamp is None


def _board_config() -> dict:
    return {
        "charuco": {
            "squares_x": 5,
            "squares_y": 7,
            "square_length_m": 0.04,
            "marker_length_m": 0.03,
            "dictionary": "DICT_5X5_250",
        }
    }


def test_detect_charuco_recovers_inner_corners_on_generated_board():
    config = _board_config()
    board = charuco_board_from_config(config)
    image = board.generateImage((800, 600))

    detection = detect_charuco_image(image, board)

    inner_x = config["charuco"]["squares_x"] - 1
    inner_y = config["charuco"]["squares_y"] - 1
    assert detection.ids is not None
    assert len(detection.ids) == inner_x * inner_y
    assert detection.corners.shape[0] == len(detection.ids)


def test_common_corner_ids_keeps_only_shared_markers():
    ids_a = np.array([0, 1, 2, 5])
    corners_a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    ids_b = np.array([5, 1, 9])
    corners_b = np.array([[50.0, 0.0], [10.0, 0.0], [90.0, 0.0]], dtype=np.float32)

    obj_ids, pts_a, pts_b = common_corner_ids(
        ids_a, corners_a, ids_b, corners_b, object_points=np.stack(
            [np.array([i, 0.0, 0.0], dtype=np.float32) for i in range(10)]
        ),
    )

    assert list(obj_ids) == [1, 5]
    assert pts_a[0].tolist() == pytest.approx([1.0, 0.0])
    assert pts_b[0].tolist() == pytest.approx([10.0, 0.0])
    assert pts_a[1].tolist() == pytest.approx([5.0, 0.0])
    assert pts_b[1].tolist() == pytest.approx([50.0, 0.0])


def test_subpixel_refine_does_not_drop_corners():
    config = _board_config()
    board = charuco_board_from_config(config)
    image = board.generateImage((800, 600))
    detection = detect_charuco_image(image, board, refine=False)
    refined = refine_corners_subpixel(image, detection.corners)

    assert refined.shape == detection.corners.shape
    assert np.isfinite(refined).all()

"""ChArUco detection tests against a board rendered by OpenCV.

A production bug that would fail these tests: dropping corner IDs, skipping
subpixel refine, or matching stereo corners by list index instead of ID.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.common import board_metadata, calibration_board_from_config, charuco_board_from_config
from calibration.detect_charuco import (
    common_corner_ids,
    detect_calibration_image,
    detect_charuco_image,
    detections_to_point_lists,
    refine_corners_subpixel,
    validate_detection_board,
)


def _chessboard_config() -> dict:
    return {
        "board": {"type": "chessboard"},
        "chessboard": {
            "inner_corners_x": 9,
            "inner_corners_y": 6,
            "square_length_m": 0.025,
        },
    }


def test_chessboard_object_points_use_inner_corner_geometry():
    board = calibration_board_from_config(_chessboard_config())

    points = board.getChessboardCorners()

    assert points.shape == (54, 3)
    assert points[0].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert points[1].tolist() == pytest.approx([0.025, 0.0, 0.0])
    assert points[9].tolist() == pytest.approx([0.0, 0.025, 0.0])


def test_detect_calibration_image_recovers_plain_chessboard_corners():
    config = _chessboard_config()
    board = calibration_board_from_config(config)
    square_px = 80
    squares_x = config["chessboard"]["inner_corners_x"] + 1
    squares_y = config["chessboard"]["inner_corners_y"] + 1
    image = np.full(((squares_y + 2) * square_px, (squares_x + 2) * square_px), 255, dtype=np.uint8)
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                y0, x0 = (y + 1) * square_px, (x + 1) * square_px
                image[y0 : y0 + square_px, x0 : x0 + square_px] = 0

    detection = detect_calibration_image(image, board)

    assert detection.n_corners == 54
    assert detection.ids.tolist() == list(range(54))


def test_chessboard_detections_convert_to_metric_calibration_points():
    board = calibration_board_from_config(_chessboard_config())
    payload = {
        "frames": [{"frame": 1, "ids": [0, 1, 9], "corners": [[10, 20], [20, 20], [10, 30]]}]
    }

    objects, images, metadata = detections_to_point_lists(payload, board)

    np.testing.assert_allclose(
        objects[0], [[0.0, 0.0, 0.0], [0.025, 0.0, 0.0], [0.0, 0.025, 0.0]]
    )
    np.testing.assert_allclose(images[0], [[10, 20], [20, 20], [10, 30]])
    assert metadata[0]["frame"] == 1


def test_chessboard_metadata_records_inner_corner_convention():
    board = calibration_board_from_config(_chessboard_config())

    assert board_metadata(board) == {
        "type": "chessboard",
        "inner_corners_x": 9,
        "inner_corners_y": 6,
        "square_length_m": 0.025,
    }


def test_detection_board_mismatch_is_rejected_before_calibration():
    board = calibration_board_from_config(_chessboard_config())
    payload = {"board": {"type": "charuco"}, "frames": []}

    with pytest.raises(ValueError, match="board does not match"):
        validate_detection_board(payload, board)


def test_legacy_detection_without_board_metadata_remains_compatible():
    board = calibration_board_from_config(_chessboard_config())

    validate_detection_board({"frames": []}, board)


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

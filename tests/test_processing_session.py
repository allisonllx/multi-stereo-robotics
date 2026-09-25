from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from processing.session import associate_nearest, decode_padded_buffer, load_frames, load_numeric_csv


def test_decode_padded_float_buffer_removes_row_padding(tmp_path: Path):
    rows = np.array([[1, 2, 3, 99], [4, 5, 6, 99]], dtype=np.float32)
    path = tmp_path / "depth.bin"
    path.write_bytes(rows.tobytes())
    decoded = decode_padded_buffer(path, width=3, height=2, bytes_per_row=16, dtype=np.float32)
    np.testing.assert_array_equal(decoded, [[1, 2, 3], [4, 5, 6]])


def test_associate_nearest_reports_signed_delta_and_rejects_stale():
    result = associate_nearest(np.array([0.1, 1.0]), np.array([0.0, 0.25]), max_delta_s=0.2)
    np.testing.assert_array_equal(result.indices, [0, -1])
    assert result.delta_s[0] == pytest.approx(-0.1)
    assert np.isnan(result.delta_s[1])


def test_load_frames_parses_row_major_camera_pose(tmp_path: Path):
    fields = ["frame_id", "timestamp_s", "image_path", "fx", "fy", "cx", "cy", "tracking_state"]
    fields += [f"pose_m{r}{c}" for r in range(4) for c in range(4)]
    pose = np.eye(4); pose[0, 3] = 1.25
    with (tmp_path / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        row = {"frame_id": 0, "timestamp_s": 2.0, "image_path": "rgb/000000.jpg", "fx": 10, "fy": 11, "cx": 5, "cy": 6, "tracking_state": "normal"}
        row.update({f"pose_m{r}{c}": pose[r, c] for r in range(4) for c in range(4)})
        writer.writerow(row)
    (tmp_path / "rgb").mkdir(); (tmp_path / "rgb/000000.jpg").write_bytes(b"x")
    frames = load_frames(tmp_path)
    assert len(frames) == 1
    np.testing.assert_allclose(frames[0].camera_to_world, pose)
    np.testing.assert_allclose(frames[0].intrinsics, [[10, 0, 5], [0, 11, 6], [0, 0, 1]])


def test_empty_sensor_csv_retains_column_shape(tmp_path: Path):
    path = tmp_path / "location.csv"
    path.write_text("unix_time_s,latitude,longitude\n", encoding="utf-8")
    names, values = load_numeric_csv(path)
    assert names == ["unix_time_s", "latitude", "longitude"]
    assert values.shape == (0, 3)

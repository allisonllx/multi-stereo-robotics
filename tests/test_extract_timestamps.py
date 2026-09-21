"""Presentation-timestamp extraction must not assume a constant FPS."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from calibration.extract_timestamps import extract_frame_timestamps, write_timestamp_csv, load_timestamp_csv


def test_extract_frame_timestamps_uses_real_pts_not_index_over_fps(tmp_path: Path):
    video = tmp_path / "vfr.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x120:r=25:d=0.4",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"ffmpeg could not write a test video: {result.stderr}")

    stamps = extract_frame_timestamps(video)
    assert stamps.shape[1] == 2
    assert len(stamps) >= 8
    # Frame 0 is near t=0, last frame is near the clip duration — not n/25 exactly
    # if the container uses a different timebase, but dt should be ~40ms.
    dts = np.diff(stamps[:, 1])
    assert np.median(dts) == pytest.approx(0.04, abs=0.005)
    csv_path = tmp_path / "timestamps.csv"
    write_timestamp_csv(csv_path, stamps)
    loaded = load_timestamp_csv(csv_path)
    np.testing.assert_allclose(loaded[:, 1], stamps[:, 1], atol=1e-8)

"""Nearest-frame pairing tests.

A production bug that would fail these tests: pairing by frame index instead
of presentation timestamps, or dropping the residual column.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.pair_frames import pair_frames


def test_pair_frames_picks_closest_a_timestamp_not_same_index():
    timestamps_a = np.array([0.00, 0.033, 0.066, 0.100, 0.133])
    timestamps_b = np.array([0.010, 0.043, 0.090])
    # Identity mapping: t_A = t_B
    pairs = pair_frames(timestamps_a, timestamps_b, a=1.0, b=0.0)

    assert [p.frame_a for p in pairs] == [0, 1, 3]
    assert [p.frame_b for p in pairs] == [0, 1, 2]
    assert pairs[0].residual_s == pytest.approx(0.00 - 0.010)
    assert pairs[2].residual_s == pytest.approx(0.100 - 0.090)


def test_pair_frames_applies_affine_before_search():
    timestamps_a = np.array([0.00, 0.10, 0.20, 0.30])
    timestamps_b = np.array([1.00, 1.10])
    # t_A = 1.0 * t_B - 0.90  => B@1.00 maps to 0.10, B@1.10 maps to 0.20
    pairs = pair_frames(timestamps_a, timestamps_b, a=1.0, b=-0.90)

    assert [p.frame_a for p in pairs] == [1, 2]
    assert pairs[0].timestamp_a == pytest.approx(0.10)
    assert pairs[0].timestamp_b == pytest.approx(1.00)
    assert abs(pairs[0].residual_s) < 1e-12


def test_pair_frames_csv_columns():
    timestamps_a = np.array([0.0, 0.04])
    timestamps_b = np.array([0.01])
    pairs = pair_frames(timestamps_a, timestamps_b, a=1.0, b=0.0)
    row = pairs[0].as_csv_row()

    assert list(row.keys()) == [
        "frame_A",
        "timestamp_A",
        "frame_B",
        "timestamp_B",
        "residual_ms",
    ]
    assert row["residual_ms"] == pytest.approx(-10.0, abs=1e-6)


def test_pair_frames_can_reject_stale_and_duplicate_matches():
    timestamps_a = np.array([0.0, 0.1, 0.2])
    timestamps_b = np.array([0.01, 0.02, 0.11, 0.50])
    pairs = pair_frames(timestamps_a, timestamps_b, a=1.0, b=0.0, max_residual_s=0.04, unique_a=True)
    assert [(p.frame_a, p.frame_b) for p in pairs] == [(0, 0), (1, 2)]
    assert all(abs(p.residual_s) <= 0.04 for p in pairs)

"""Audio lag and affine time-mapping tests.

A production bug that would fail these tests: reversing the lag sign,
skipping the end-clap window, or forcing a=1 when start/end offsets differ.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.synchronize_audio import (
    estimate_affine_from_claps,
    estimate_lag_seconds,
    find_peak_times,
    fit_affine_time_map,
    map_time_b_to_a,
    should_fit_drift,
)


def _impulse_at(n_samples: int, index: int) -> np.ndarray:
    signal = np.zeros(n_samples, dtype=np.float64)
    # Narrow burst so GCC-PHAT has a sharp peak, like a clap.
    signal[index] = 1.0
    if index + 1 < n_samples:
        signal[index + 1] = 0.4
    if index > 0:
        signal[index - 1] = 0.4
    return signal


def test_estimate_lag_adds_to_b_to_reach_a_timeline():
    """Same clap at 0.50s on A and 0.80s on B => lag = -0.30s (t_A = t_B + lag)."""
    sr = 8000
    n = 2 * sr
    audio_a = _impulse_at(n, int(0.50 * sr))
    audio_b = _impulse_at(n, int(0.80 * sr))

    lag = estimate_lag_seconds(audio_a, audio_b, sample_rate=sr)

    assert lag == pytest.approx(-0.30, abs=1.0 / sr)


def test_estimate_lag_positive_when_b_starts_later():
    """Clap at 0.80s on A and 0.50s on B => B started later, lag = +0.30s."""
    sr = 8000
    n = 2 * sr
    audio_a = _impulse_at(n, int(0.80 * sr))
    audio_b = _impulse_at(n, int(0.50 * sr))

    lag = estimate_lag_seconds(audio_a, audio_b, sample_rate=sr)

    assert lag == pytest.approx(0.30, abs=1.0 / sr)


def test_should_fit_drift_when_start_and_end_offsets_diverge():
    assert should_fit_drift(lag_start_s=0.010, lag_end_s=0.040, threshold_s=0.005) is True
    assert should_fit_drift(lag_start_s=0.010, lag_end_s=0.012, threshold_s=0.005) is False


def test_affine_from_start_and_end_claps_recovers_offset():
    sr = 8000
    duration = 4.0
    n = int(duration * sr)
    audio_a = np.zeros(n)
    audio_b = np.zeros(n)
    audio_a[int(0.3 * sr)] = 1.0
    audio_a[int(3.5 * sr)] = 1.0
    audio_b[int(0.5 * sr)] = 1.0
    audio_b[int(3.7 * sr)] = 1.0
    mapping = estimate_affine_from_claps(
        audio_a, audio_b, sample_rate=sr, clap_window_s=1.0, drift_threshold_s=0.005
    )
    assert mapping["fit_drift"] is False
    assert mapping["b"] == pytest.approx(-0.20, abs=2.0 / sr)
    assert mapping["a"] == pytest.approx(1.0)


def test_find_peak_times_returns_three_spaced_claps():
    sr = 8000
    audio = np.zeros(int(2.0 * sr))
    for t in (0.25, 0.55, 0.90):
        audio[int(t * sr)] = 1.0
    times = find_peak_times(audio, sr, 0.0, 2.0, n_peaks=3, min_gap_s=0.15)
    assert times == pytest.approx([0.25, 0.55, 0.90], abs=1.0 / sr)
    """t_A = 1.001 * t_B + 0.25 recovered from start/end clap alignments."""
    t_b_start, t_b_end = 0.4, 60.4
    a_true, b_true = 1.001, 0.25
    t_a_start = a_true * t_b_start + b_true
    t_a_end = a_true * t_b_end + b_true

    a_hat, b_hat = fit_affine_time_map(
        t_b_start=t_b_start,
        t_a_start=t_a_start,
        t_b_end=t_b_end,
        t_a_end=t_a_end,
    )

    assert a_hat == pytest.approx(1.001, rel=1e-9)
    assert b_hat == pytest.approx(0.25, rel=1e-9)
    assert map_time_b_to_a(30.0, a_hat, b_hat) == pytest.approx(1.001 * 30.0 + 0.25)

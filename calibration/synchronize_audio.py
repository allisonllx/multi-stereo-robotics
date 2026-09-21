"""Synchronize two phone recordings from clap audio.

Convention: Camera B timestamps map onto Camera A's timeline by

    t_A = a * t_B + b

`b` is the initial offset (seconds to add to B at t_B=0). `a` corrects
clock-rate drift. When start/end clap offsets agree within the configured
threshold, `a` is left at 1 and `b` is the start-window lag.

Lag sign: the same physical clap at t_A and t_B implies lag = t_A - t_B,
i.e. t_A = t_B + lag when a=1.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.io import wavfile

from calibration.common import load_config, output_dir, resolve_path, save_yaml


def estimate_lag_seconds(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sample_rate: float,
    max_lag_s: float | None = None,
) -> float:
    """GCC-PHAT lag such that t_A = t_B + lag.

    If the same event is at sample i_A in A and i_B in B, lag = (i_A - i_B) / fs.
    """
    sig_a = np.asarray(audio_a, dtype=np.float64).ravel()
    sig_b = np.asarray(audio_b, dtype=np.float64).ravel()
    if sig_a.size == 0 or sig_b.size == 0:
        raise ValueError("Audio segments must be non-empty")

    n = int(2 ** np.ceil(np.log2(sig_a.size + sig_b.size)))
    spectrum_a = np.fft.rfft(sig_a, n=n)
    spectrum_b = np.fft.rfft(sig_b, n=n)
    cross = spectrum_a * np.conj(spectrum_b)
    cross /= np.abs(cross) + 1e-15
    correlation = np.fft.irfft(cross, n=n)

    max_lag_samples = int(min(sig_a.size, sig_b.size) - 1)
    if max_lag_s is not None:
        max_lag_samples = min(max_lag_samples, int(max_lag_s * sample_rate))
    if max_lag_samples < 1:
        raise ValueError("Audio segments are too short to estimate a lag")

    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    wrapped = np.concatenate((correlation[-max_lag_samples:], correlation[: max_lag_samples + 1]))
    peak = int(lags[int(np.argmax(wrapped))])
    return float(peak) / float(sample_rate)


def should_fit_drift(lag_start_s: float, lag_end_s: float, threshold_s: float) -> bool:
    return abs(lag_end_s - lag_start_s) > threshold_s


def fit_affine_time_map(
    t_b_start: float,
    t_a_start: float,
    t_b_end: float,
    t_a_end: float,
) -> tuple[float, float]:
    """Fit t_A = a t_B + b from two aligned clap events."""
    dt_b = t_b_end - t_b_start
    if abs(dt_b) < 1e-9:
        raise ValueError("Start and end clap times on camera B are identical")
    a = (t_a_end - t_a_start) / dt_b
    b = t_a_start - a * t_b_start
    return float(a), float(b)


def map_time_b_to_a(t_b: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    return a * t_b + b


def _slice_audio(audio: np.ndarray, sample_rate: float, t0: float, t1: float) -> np.ndarray:
    i0 = max(0, int(round(t0 * sample_rate)))
    i1 = min(audio.size, int(round(t1 * sample_rate)))
    if i1 <= i0:
        raise ValueError(f"Empty audio slice [{t0}, {t1}]")
    return audio[i0:i1]


def estimate_window_lag(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sample_rate: float,
    t0_a: float,
    t1_a: float,
    t0_b: float,
    t1_b: float,
) -> float:
    """Lag relating the two cameras at a local clap window.

    Segments are sliced in each camera's own timeline, so the returned value
    already includes the difference in window start times:

        t_A = t_B + lag
    """
    seg_a = _slice_audio(audio_a, sample_rate, t0_a, t1_a)
    seg_b = _slice_audio(audio_b, sample_rate, t0_b, t1_b)
    lag_in_window = estimate_lag_seconds(seg_a, seg_b, sample_rate)
    return (t0_a - t0_b) + lag_in_window


def extract_audio_mono(video_path: str | Path, sample_rate: int = 48000) -> tuple[np.ndarray, int]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            str(wav_path),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {video_path}:\n{result.stderr}")
        rate, data = wavfile.read(wav_path)
    finally:
        wav_path.unlink(missing_ok=True)

    samples = np.asarray(data, dtype=np.float64)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak
    return samples, int(rate)


def find_peak_times(
    audio: np.ndarray,
    sample_rate: float,
    t0: float,
    t1: float,
    n_peaks: int = 3,
    min_gap_s: float = 0.15,
) -> list[float]:
    """Return the strongest clap-like peaks in [t0, t1], in seconds."""
    from scipy.signal import find_peaks

    seg = _slice_audio(audio, sample_rate, t0, t1)
    envelope = np.abs(seg)
    min_gap = max(1, int(min_gap_s * sample_rate))
    peaks, _ = find_peaks(envelope, distance=min_gap)
    if peaks.size == 0:
        return []
    strongest = np.argsort(envelope[peaks])[::-1][:n_peaks]
    return sorted(float(t0 + peaks[i] / sample_rate) for i in strongest)


def _duration_s(audio: np.ndarray, sample_rate: float) -> float:
    return float(audio.size) / float(sample_rate)


def estimate_affine_from_claps(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sample_rate: float,
    clap_window_s: float = 8.0,
    drift_threshold_s: float = 0.005,
) -> dict:
    duration_a = _duration_s(audio_a, sample_rate)
    duration_b = _duration_s(audio_b, sample_rate)
    window = min(clap_window_s, 0.45 * duration_a, 0.45 * duration_b)
    if window < 0.2:
        raise ValueError("Recordings are too short for start/end clap windows")

    lag_start = estimate_window_lag(
        audio_a, audio_b, sample_rate, 0.0, window, 0.0, window
    )
    t0_a_end = duration_a - window
    t0_b_end = duration_b - window
    lag_end = estimate_window_lag(
        audio_a,
        audio_b,
        sample_rate,
        t0_a_end,
        duration_a,
        t0_b_end,
        duration_b,
    )

    t_b_start = 0.5 * window
    t_b_end = t0_b_end + 0.5 * window
    t_a_start = t_b_start + lag_start
    t_a_end = t_b_end + lag_end

    fit_drift = should_fit_drift(lag_start, lag_end, drift_threshold_s)
    if fit_drift:
        a, b = fit_affine_time_map(t_b_start, t_a_start, t_b_end, t_a_end)
    else:
        a, b = 1.0, float(lag_start)

    mapping = {
        "a": float(a),
        "b": float(b),
        "lag_start_s": float(lag_start),
        "lag_end_s": float(lag_end),
        "drift_s": float(lag_end - lag_start),
        "fit_drift": bool(fit_drift),
        "window_s": float(window),
        "duration_a_s": duration_a,
        "duration_b_s": duration_b,
        "sample_rate": float(sample_rate),
        "t_b_start_s": float(t_b_start),
        "t_b_end_s": float(t_b_end),
        "clap_times_start_A_s": find_peak_times(audio_a, sample_rate, 0.0, window),
        "clap_times_end_A_s": find_peak_times(audio_a, sample_rate, t0_a_end, duration_a),
        "clap_times_start_B_s": find_peak_times(audio_b, sample_rate, 0.0, window),
        "clap_times_end_B_s": find_peak_times(audio_b, sample_rate, t0_b_end, duration_b),
    }
    return mapping


def synchronize_videos(
    video_a: str | Path,
    video_b: str | Path,
    clap_window_s: float = 8.0,
    drift_threshold_s: float = 0.005,
    sample_rate: int = 48000,
) -> dict[str, float | bool | str]:
    audio_a, rate_a = extract_audio_mono(video_a, sample_rate=sample_rate)
    audio_b, rate_b = extract_audio_mono(video_b, sample_rate=sample_rate)
    if rate_a != rate_b:
        raise RuntimeError(f"Audio sample rates differ: {rate_a} vs {rate_b}")
    mapping = estimate_affine_from_claps(
        audio_a,
        audio_b,
        sample_rate=rate_a,
        clap_window_s=clap_window_s,
        drift_threshold_s=drift_threshold_s,
    )
    mapping["video_a"] = str(video_a)
    mapping["video_b"] = str(video_b)
    mapping["model"] = "t_A = a * t_B + b"
    return mapping


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cross-correlate clap audio and fit t_A = a t_B + b"
    )
    parser.add_argument("--config", default=None, help="Path to calibration/config.yaml")
    parser.add_argument("--video-a", default=None)
    parser.add_argument("--video-b", default=None)
    parser.add_argument("--output", default=None, help="JSON mapping path")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    sync_cfg = config.get("sync", {})
    cameras = config["cameras"]
    video_a = args.video_a or resolve_path(config, cameras["A"]["video"])
    video_b = args.video_b or resolve_path(config, cameras["B"]["video"])
    mapping = synchronize_videos(
        video_a,
        video_b,
        clap_window_s=float(sync_cfg.get("clap_window_s", 8.0)),
        drift_threshold_s=float(sync_cfg.get("drift_threshold_ms", 5.0)) / 1000.0,
        sample_rate=int(sync_cfg.get("audio_sample_rate", 48000)),
    )
    out_dir = output_dir(config)
    output_path = Path(args.output) if args.output else out_dir / "time_mapping.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    save_yaml(out_dir / "time_mapping.yaml", mapping)
    print(f"Wrote {output_path}")
    print(
        f"t_A = {mapping['a']:.9f} * t_B + {mapping['b']:.6f}  "
        f"(start lag {mapping['lag_start_s']*1000:.2f} ms, "
        f"end lag {mapping['lag_end_s']*1000:.2f} ms, "
        f"drift {mapping['drift_s']*1000:.2f} ms)"
    )
    print(f"start claps A (s): {mapping.get('clap_times_start_A_s')}")
    print(f"start claps B (s): {mapping.get('clap_times_start_B_s')}")
    print(f"end claps A (s):   {mapping.get('clap_times_end_A_s')}")
    print(f"end claps B (s):   {mapping.get('clap_times_end_B_s')}")


if __name__ == "__main__":
    main()

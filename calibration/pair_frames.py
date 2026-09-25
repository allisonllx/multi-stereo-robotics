"""Pair Camera B frames to the nearest Camera A frame on A's timeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from calibration.common import load_config, load_yaml, output_dir, resolve_path
from calibration.extract_timestamps import extract_frame_timestamps, load_timestamp_csv
from calibration.synchronize_audio import map_time_b_to_a


@dataclass(frozen=True)
class FramePair:
    frame_a: int
    timestamp_a: float
    frame_b: int
    timestamp_b: float
    residual_s: float

    def as_csv_row(self) -> dict[str, float | int]:
        return {
            "frame_A": int(self.frame_a),
            "timestamp_A": float(self.timestamp_a),
            "frame_B": int(self.frame_b),
            "timestamp_B": float(self.timestamp_b),
            "residual_ms": float(self.residual_s) * 1000.0,
        }


def pair_frames(
    timestamps_a: np.ndarray,
    timestamps_b: np.ndarray,
    a: float,
    b: float,
    max_residual_s: float | None = None,
    unique_a: bool = False,
) -> list[FramePair]:
    """Map each B timestamp into A's timeline and take the nearest A frame.

    residual_s = timestamp_A - (a * timestamp_B + b)
    """
    times_a = np.asarray(timestamps_a, dtype=np.float64).ravel()
    times_b = np.asarray(timestamps_b, dtype=np.float64).ravel()
    if times_a.size == 0 or times_b.size == 0:
        return []

    mapped = np.asarray(map_time_b_to_a(times_b, a, b), dtype=np.float64)
    pairs: list[FramePair] = []
    for frame_b, (t_b, t_mapped) in enumerate(zip(times_b, mapped)):
        frame_a = int(np.argmin(np.abs(times_a - t_mapped)))
        t_a = float(times_a[frame_a])
        pair = (
            FramePair(
                frame_a=frame_a,
                timestamp_a=t_a,
                frame_b=frame_b,
                timestamp_b=float(t_b),
                residual_s=t_a - float(t_mapped),
            )
        )
        if max_residual_s is None or abs(pair.residual_s) <= max_residual_s:
            pairs.append(pair)
    if unique_a:
        best_by_a: dict[int, FramePair] = {}
        for pair in pairs:
            current = best_by_a.get(pair.frame_a)
            if current is None or abs(pair.residual_s) < abs(current.residual_s):
                best_by_a[pair.frame_a] = pair
        pairs = sorted(best_by_a.values(), key=lambda value: value.frame_b)
    return pairs


def write_pairs_csv(path: str | Path, pairs: list[FramePair]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_A", "timestamp_A", "frame_B", "timestamp_B", "residual_ms"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair.as_csv_row())


def load_pairs_csv(path: str | Path) -> list[FramePair]:
    pairs: list[FramePair] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pairs.append(
                FramePair(
                    frame_a=int(row["frame_A"]),
                    timestamp_a=float(row["timestamp_A"]),
                    frame_b=int(row["frame_B"]),
                    timestamp_b=float(row["timestamp_B"]),
                    residual_s=float(row["residual_ms"]) / 1000.0,
                )
            )
    return pairs


def _load_mapping(path: Path) -> tuple[float, float, dict]:
    if path.suffix.lower() == ".json":
        mapping = json.loads(path.read_text(encoding="utf-8"))
    else:
        mapping = load_yaml(path)
    return float(mapping["a"]), float(mapping["b"]), mapping


def _nearest_index(timestamps: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(timestamps - t)))


def _print_clap_frame_check(
    times_a: np.ndarray,
    times_b: np.ndarray,
    a: float,
    b: float,
    mapping: dict,
) -> None:
    """Print nearest frames for detected claps so they can be checked by eye."""
    groups = (
        ("start A", mapping.get("clap_times_start_A_s") or [], True),
        ("start B", mapping.get("clap_times_start_B_s") or [], False),
        ("end A", mapping.get("clap_times_end_A_s") or [], True),
        ("end B", mapping.get("clap_times_end_B_s") or [], False),
    )
    if not any(times for _label, times, _is_a in groups):
        return
    print("clap frame check (open these frames and confirm the hands meet):")
    for label, times, is_a in groups:
        for t in times:
            if is_a:
                frame_a = _nearest_index(times_a, t)
                t_b = (t - b) / a if a != 0 else t
                frame_b = _nearest_index(times_b, t_b)
            else:
                frame_b = _nearest_index(times_b, t)
                t_a = a * t + b
                frame_a = _nearest_index(times_a, t_a)
            print(
                f"  {label} t={t:.3f}s -> frame_A={frame_a} ({times_a[frame_a]:.3f}s), "
                f"frame_B={frame_b} ({times_b[frame_b]:.3f}s)"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pair each Camera B frame with the closest Camera A frame"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--timestamps-a", default=None)
    parser.add_argument("--timestamps-b", default=None)
    parser.add_argument("--mapping", default=None, help="time_mapping.json from synchronize_audio")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = output_dir(config)
    mapping_path = Path(args.mapping) if args.mapping else out_dir / "time_mapping.json"
    a, b, mapping = _load_mapping(mapping_path)

    ts_a_path = Path(args.timestamps_a) if args.timestamps_a else out_dir / "timestamps_A.csv"
    ts_b_path = Path(args.timestamps_b) if args.timestamps_b else out_dir / "timestamps_B.csv"
    if ts_a_path.exists() and ts_b_path.exists():
        stamps_a = load_timestamp_csv(ts_a_path)
        stamps_b = load_timestamp_csv(ts_b_path)
        times_a = stamps_a[:, 1]
        times_b = stamps_b[:, 1]
    else:
        cameras = config["cameras"]
        times_a = extract_frame_timestamps(resolve_path(config, cameras["A"]["video"]))[:, 1]
        times_b = extract_frame_timestamps(resolve_path(config, cameras["B"]["video"]))[:, 1]

    max_residual_s = float(config.get("sync", {}).get("max_pair_residual_ms", 0)) / 1000.0
    pairs = pair_frames(times_a, times_b, a=a, b=b,
                        max_residual_s=max_residual_s if max_residual_s > 0 else None,
                        unique_a=True)
    output_path = Path(args.output) if args.output else out_dir / "synchronization.csv"
    write_pairs_csv(output_path, pairs)

    residuals_ms = np.array([p.residual_s * 1000.0 for p in pairs], dtype=np.float64)
    max_abs = float(np.max(np.abs(residuals_ms))) if residuals_ms.size else float("nan")
    mean_abs = float(np.mean(np.abs(residuals_ms))) if residuals_ms.size else float("nan")
    print(f"Wrote {output_path} ({len(pairs)} pairs)")
    print(f"mean |residual| = {mean_abs:.3f} ms, max |residual| = {max_abs:.3f} ms")
    print(f"mapping: t_A = {a:.9f} * t_B + {b:.6f}")
    if mapping.get("fit_drift"):
        print("affine drift correction was applied")
    else:
        print("clock drift was below threshold; used a=1")
    _print_clap_frame_check(times_a, times_b, a, b, mapping)


if __name__ == "__main__":
    main()

"""Read the actual presentation timestamp of every video frame.

Do not assume frame i occurred at i/FPS. Phone recordings are often VFR.
Prefer ffprobe `best_effort_timestamp_time`; fall back to OpenCV POS_MSEC.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from calibration.common import load_config, output_dir, resolve_path


def extract_frame_timestamps_ffprobe(video_path: str | Path) -> np.ndarray:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time,pts_time",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}:\n{result.stderr}")

    times: list[float] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        value = None
        for part in parts:
            if part and part.upper() != "N/A":
                value = float(part)
                break
        if value is not None:
            times.append(value)

    if not times:
        raise RuntimeError(f"ffprobe returned no frame timestamps for {video_path}")
    indices = np.arange(len(times), dtype=np.int32)
    return np.column_stack((indices, np.asarray(times, dtype=np.float64)))


def extract_frame_timestamps_opencv(video_path: str | Path) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video {video_path}")
    rows: list[tuple[int, float]] = []
    index = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        msec = capture.get(cv2.CAP_PROP_POS_MSEC)
        rows.append((index, float(msec) / 1000.0))
        index += 1
    capture.release()
    if not rows:
        raise RuntimeError(f"OpenCV read zero frames from {video_path}")
    return np.asarray(rows, dtype=np.float64)


def extract_frame_timestamps(video_path: str | Path) -> np.ndarray:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    try:
        return extract_frame_timestamps_ffprobe(video_path)
    except (FileNotFoundError, RuntimeError):
        return extract_frame_timestamps_opencv(video_path)


def write_timestamp_csv(path: str | Path, stamps: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "timestamp_s"])
        for frame, timestamp in stamps:
            writer.writerow([int(frame), f"{float(timestamp):.9f}"])


def load_timestamp_csv(path: str | Path) -> np.ndarray:
    rows: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([float(row["frame"]), float(row["timestamp_s"])])
    return np.asarray(rows, dtype=np.float64)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract per-frame presentation timestamps")
    parser.add_argument("--config", default=None)
    parser.add_argument("--video", default=None, help="Override; otherwise both cameras are processed")
    parser.add_argument("--camera", choices=("A", "B"), default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = output_dir(config)

    jobs: list[tuple[str, Path, Path]] = []
    if args.video:
        camera = args.camera or "A"
        output = Path(args.output) if args.output else out_dir / f"timestamps_{camera}.csv"
        jobs.append((camera, Path(args.video), output))
    else:
        for camera in ("A", "B"):
            if args.camera and camera != args.camera:
                continue
            video = resolve_path(config, config["cameras"][camera]["video"])
            output = out_dir / f"timestamps_{camera}.csv"
            jobs.append((camera, video, output))

    for camera, video, output in jobs:
        stamps = extract_frame_timestamps(video)
        write_timestamp_csv(output, stamps)
        dts = np.diff(stamps[:, 1])
        median_dt = float(np.median(dts)) if dts.size else float("nan")
        print(
            f"{camera}: {len(stamps)} frames from {video} -> {output} "
            f"(median dt {median_dt*1000:.3f} ms)"
        )


if __name__ == "__main__":
    main()

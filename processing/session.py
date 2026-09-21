from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp_s: float
    image_path: Path
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    tracking_state: str
    values: dict[str, str]


@dataclass(frozen=True)
class Association:
    indices: np.ndarray
    delta_s: np.ndarray


def decode_padded_buffer(path: str | Path, width: int, height: int, bytes_per_row: int, dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if width <= 0 or height <= 0 or bytes_per_row < width * dtype.itemsize:
        raise ValueError("invalid buffer dimensions or row stride")
    raw = Path(path).read_bytes()
    expected = height * bytes_per_row
    if len(raw) != expected:
        raise ValueError(f"buffer has {len(raw)} bytes; expected {expected}")
    values_per_row = bytes_per_row // dtype.itemsize
    return np.frombuffer(raw, dtype=dtype).reshape(height, values_per_row)[:, :width].copy()


def associate_nearest(query_times, sample_times, max_delta_s: float | None = None) -> Association:
    query = np.asarray(query_times, dtype=np.float64)
    samples = np.asarray(sample_times, dtype=np.float64)
    if samples.size == 0:
        return Association(np.full(query.shape, -1, dtype=int), np.full(query.shape, np.nan))
    if np.any(np.diff(samples) < 0):
        raise ValueError("sample timestamps must be sorted")
    right = np.searchsorted(samples, query, side="left")
    right = np.clip(right, 0, samples.size - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(samples[left] - query) <= np.abs(samples[right] - query)
    indices = np.where(use_left, left, right).astype(int)
    delta = samples[indices] - query
    if max_delta_s is not None:
        stale = np.abs(delta) > max_delta_s
        indices[stale] = -1
        delta[stale] = np.nan
    return Association(indices, delta)


def load_frames(session_dir: str | Path) -> list[FrameRecord]:
    root = Path(session_dir).resolve()
    with (root / "frames.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    frames: list[FrameRecord] = []
    previous = -np.inf
    for expected_id, row in enumerate(rows):
        frame_id = int(row["frame_id"])
        timestamp = float(row["timestamp_s"])
        if frame_id != expected_id or timestamp <= previous:
            raise ValueError("frames must have contiguous IDs and strictly increasing timestamps")
        previous = timestamp
        image_path = root / row["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        K = np.array([[float(row["fx"]), 0, float(row["cx"])], [0, float(row["fy"]), float(row["cy"])], [0, 0, 1]], dtype=np.float64)
        pose_values = [float(row[f"pose_m{r}{c}"]) for r in range(4) for c in range(4)]
        pose = np.asarray(pose_values, dtype=np.float64).reshape(4, 4)
        frames.append(FrameRecord(frame_id, timestamp, image_path, K, pose, row["tracking_state"], row))
    return frames


def load_numeric_csv(path: str | Path) -> tuple[list[str], np.ndarray]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        names = list(reader.fieldnames or [])
        data = [[float(row[name]) for name in names] for row in reader]
    return names, np.asarray(data, dtype=np.float64)


def write_frame_sensor_associations(session_dir: str | Path, output_csv: str | Path,
                                    max_location_age_s: float = 2.0,
                                    max_heading_age_s: float = 0.5) -> Path:
    root = Path(session_dir); frames = load_frames(root)
    streams = []
    for prefix, filename, max_age in (("gps", "location.csv", max_location_age_s), ("heading", "heading.csv", max_heading_age_s)):
        names, values = load_numeric_csv(root / filename)
        if "unix_time_s" not in names: raise ValueError(f"{filename} has no unix_time_s column")
        streams.append((prefix, names, values, associate_nearest(
            np.array([float(f.values["unix_time_s"]) for f in frames]),
            values[:, names.index("unix_time_s")], max_age)))
    fieldnames = ["frame_id", "timestamp_s", "unix_time_s"]
    for prefix, names, _values, _association in streams:
        fieldnames += [f"{prefix}_age_s"] + [f"{prefix}_{name}" for name in names if name != "unix_time_s"]
    output = Path(output_csv); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader()
        for index, frame in enumerate(frames):
            row = {"frame_id": frame.frame_id, "timestamp_s": frame.timestamp_s, "unix_time_s": frame.values["unix_time_s"]}
            for prefix, names, values, association in streams:
                sample_index = association.indices[index]
                row[f"{prefix}_age_s"] = "" if sample_index < 0 else abs(float(association.delta_s[index]))
                for name in names:
                    if name != "unix_time_s": row[f"{prefix}_{name}"] = "" if sample_index < 0 else values[sample_index, names.index(name)]
            writer.writerow(row)
    return output

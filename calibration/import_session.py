"""Validate an exported ARKit recording and expose it to the calibration pipeline."""

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

from calibration.common import load_config, output_dir
from calibration.extract_timestamps import write_timestamp_csv


REQUIRED_FRAME_COLUMNS = {
    "frame_id",
    "timestamp_s",
    "image_path",
    "fx",
    "fy",
    "cx",
    "cy",
    "tracking_state",
}


@dataclass(frozen=True)
class RecordingSession:
    root: Path
    manifest: dict
    frames: tuple[dict, ...]
    timestamps: np.ndarray


@dataclass(frozen=True)
class ImportResult:
    camera: str
    session_id: str
    frame_count: int
    median_interval_s: float
    rgb_dir: Path
    timestamp_path: Path
    report_path: Path


def _resolve_stream_path(root: Path, value: str, label: str) -> Path | None:
    if not value:
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a safe path relative to the session root: {value}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def load_recording_session(session_dir: str | Path) -> RecordingSession:
    root = Path(session_dir).resolve()
    manifest_path = root / "manifest.json"
    frames_path = root / "frames.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not frames_path.is_file():
        raise FileNotFoundError(f"Missing frame table: {frames_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version: {manifest.get('schema_version')!r}")
    if manifest.get("rig_role") not in {"A", "B"}:
        raise ValueError("manifest rig_role must be A or B")
    if not manifest.get("session_id"):
        raise ValueError("manifest session_id is required")

    with frames_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FRAME_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"frames.csv is missing columns: {', '.join(sorted(missing))}")
        frames = tuple(reader)
    if not frames:
        raise ValueError("frames.csv contains no frames")

    frame_ids = np.asarray([int(row["frame_id"]) for row in frames], dtype=np.int64)
    if not np.array_equal(frame_ids, np.arange(len(frames))):
        raise ValueError("frame_id values must be contiguous and start at zero")
    timestamps = np.asarray([float(row["timestamp_s"]) for row in frames], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("frame timestamps must be finite and strictly increasing")

    for row in frames:
        _resolve_stream_path(root, row["image_path"], "RGB frame")
        _resolve_stream_path(root, row.get("depth_path", ""), "depth frame")
        _resolve_stream_path(root, row.get("confidence_path", ""), "confidence frame")
        intrinsics = [float(row[name]) for name in ("fx", "fy", "cx", "cy")]
        if not all(np.isfinite(intrinsics)) or intrinsics[0] <= 0 or intrinsics[1] <= 0:
            raise ValueError(f"Invalid intrinsics for frame {row['frame_id']}")

    return RecordingSession(root, manifest, frames, timestamps)


def import_session(
    session_dir: str | Path,
    destination: str | Path,
    camera: str | None = None,
) -> ImportResult:
    session = load_recording_session(session_dir)
    role = camera or str(session.manifest["rig_role"])
    if role not in {"A", "B"}:
        raise ValueError("camera must be A or B")
    if camera is not None and camera != session.manifest["rig_role"]:
        raise ValueError(
            f"Requested camera {camera}, but manifest rig_role is {session.manifest['rig_role']}"
        )

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp_path = destination / f"timestamps_{role}.csv"
    stamps = np.column_stack((np.arange(len(session.frames)), session.timestamps))
    write_timestamp_csv(timestamp_path, stamps)

    intervals = np.diff(session.timestamps)
    median_interval = float(np.median(intervals)) if intervals.size else float("nan")
    effective_fps = 1.0 / median_interval if median_interval > 0 else None
    report = {
        "schema_version": 1,
        "session_id": session.manifest["session_id"],
        "rig_role": role,
        "session_root": str(session.root),
        "rgb_dir": str(session.root / "rgb"),
        "frame_count": len(session.frames),
        "first_timestamp_s": float(session.timestamps[0]),
        "last_timestamp_s": float(session.timestamps[-1]),
        "median_interval_s": median_interval if np.isfinite(median_interval) else None,
        "effective_fps": effective_fps,
        "device": session.manifest.get("device", {}),
        "capture": session.manifest.get("capture", {}),
    }
    report_path = destination / f"session_{role}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return ImportResult(
        camera=role,
        session_id=str(session.manifest["session_id"]),
        frame_count=len(session.frames),
        median_interval_s=median_interval,
        rgb_dir=session.root / "rgb",
        timestamp_path=timestamp_path,
        report_path=report_path,
    )


def import_pair(
    session_a_dir: str | Path,
    session_b_dir: str | Path,
    destination: str | Path,
) -> tuple[ImportResult, ImportResult]:
    session_a = load_recording_session(session_a_dir)
    session_b = load_recording_session(session_b_dir)
    if session_a.manifest["rig_role"] != "A" or session_b.manifest["rig_role"] != "B":
        raise ValueError("paired import requires an A session followed by a B session")
    if session_a.manifest["session_id"] != session_b.manifest["session_id"]:
        raise ValueError(
            "session IDs do not match: "
            f"{session_a.manifest['session_id']!r} != {session_b.manifest['session_id']!r}"
        )
    capture_a = session_a.manifest.get("capture", {})
    capture_b = session_b.manifest.get("capture", {})
    if capture_a.get("target_saved_fps") != capture_b.get("target_saved_fps"):
        raise ValueError("saved frame rates do not match between Camera A and Camera B")
    if capture_a.get("manual_profile") != capture_b.get("manual_profile"):
        raise ValueError("manual camera profiles do not match between Camera A and Camera B")
    return (
        import_session(session_a.root, destination, camera="A"),
        import_session(session_b.root, destination, camera="B"),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate and import an ARKit recording session")
    parser.add_argument("session", help="Camera A or single exported session directory")
    parser.add_argument("session_b", nargs="?", help="Optional Camera B session directory")
    parser.add_argument("--camera", choices=("A", "B"), default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    destination = Path(args.output) if args.output else output_dir(config)
    if args.session_b:
        if args.camera:
            parser.error("--camera cannot be used when importing a pair")
        results = import_pair(args.session, args.session_b, destination)
    else:
        results = (import_session(args.session, destination, camera=args.camera),)
    for result in results:
        print(
            f"{result.camera}: imported {result.frame_count} frames from {result.session_id}; "
            f"timestamps -> {result.timestamp_path}"
        )


if __name__ == "__main__":
    main()

"""ARKit session imports preserve real timestamps and reject corrupt captures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from calibration.import_session import import_pair, import_session, load_recording_session


def _write_session(root: Path, timestamps: list[float], role: str = "A", session_id: str = "walk-001") -> Path:
    (root / "rgb").mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "session_id": session_id,
        "rig_role": role,
        "device": {"model": "iPhone17,1", "system_version": "26.0"},
        "capture": {
            "target_saved_fps": 10,
            "image_width": 1920,
            "image_height": 1440,
            "manual_profile": {
                "exposure_seconds": 0.008333333,
                "iso": 100,
                "lens_position": 0.75,
                "white_balance_temperature_k": 5000,
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fields = [
        "frame_id", "timestamp_s", "image_path", "depth_path", "confidence_path",
        "fx", "fy", "cx", "cy", "tracking_state",
    ]
    with (root / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame_id, timestamp in enumerate(timestamps):
            image = root / "rgb" / f"{frame_id:06d}.jpg"
            image.write_bytes(b"jpeg-placeholder")
            writer.writerow({
                "frame_id": frame_id,
                "timestamp_s": timestamp,
                "image_path": f"rgb/{frame_id:06d}.jpg",
                "depth_path": "",
                "confidence_path": "",
                "fx": 1000,
                "fy": 1001,
                "cx": 960,
                "cy": 720,
                "tracking_state": "normal",
            })
    return root


def test_import_session_writes_pipeline_timestamps_and_report(tmp_path: Path):
    session = _write_session(tmp_path / "phone-a", [4.2, 4.301, 4.399])
    output = tmp_path / "imported"

    result = import_session(session, output, camera="A")

    assert result.frame_count == 3
    assert result.median_interval_s == pytest.approx(0.0995)
    assert result.rgb_dir == session / "rgb"
    rows = list(csv.DictReader((output / "timestamps_A.csv").open(encoding="utf-8")))
    assert [float(row["timestamp_s"]) for row in rows] == [4.2, 4.301, 4.399]
    report = json.loads((output / "session_A.json").read_text(encoding="utf-8"))
    assert report["session_id"] == "walk-001"
    assert report["effective_fps"] == pytest.approx(1 / 0.0995)


def test_load_recording_session_rejects_non_monotonic_timestamps(tmp_path: Path):
    session = _write_session(tmp_path / "bad", [1.0, 1.1, 1.05])

    with pytest.raises(ValueError, match="strictly increasing"):
        load_recording_session(session)


def test_load_recording_session_rejects_missing_rgb_frame(tmp_path: Path):
    session = _write_session(tmp_path / "bad", [1.0])
    (session / "rgb" / "000000.jpg").unlink()

    with pytest.raises(FileNotFoundError, match="000000.jpg"):
        load_recording_session(session)


def test_import_pair_requires_matching_session_ids(tmp_path: Path):
    a = _write_session(tmp_path / "a", [0.1], role="A", session_id="walk-001")
    b = _write_session(tmp_path / "b", [0.1], role="B", session_id="walk-002")

    with pytest.raises(ValueError, match="session IDs do not match"):
        import_pair(a, b, tmp_path / "out")


def test_import_pair_writes_both_timestamp_tables(tmp_path: Path):
    a = _write_session(tmp_path / "a", [0.1, 0.2], role="A")
    b = _write_session(tmp_path / "b", [0.12, 0.22], role="B")

    results = import_pair(a, b, tmp_path / "out")

    assert [result.camera for result in results] == ["A", "B"]
    assert (tmp_path / "out" / "timestamps_A.csv").is_file()
    assert (tmp_path / "out" / "timestamps_B.csv").is_file()


def test_import_pair_rejects_different_manual_camera_profiles(tmp_path: Path):
    a = _write_session(tmp_path / "a", [0.1], role="A")
    b = _write_session(tmp_path / "b", [0.1], role="B")
    manifest_path = b / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["capture"]["manual_profile"]["iso"] = 200
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manual camera profiles do not match"):
        import_pair(a, b, tmp_path / "out")


def test_import_pair_rejects_different_saved_frame_rates(tmp_path: Path):
    a = _write_session(tmp_path / "a", [0.1], role="A")
    b = _write_session(tmp_path / "b", [0.1], role="B")
    manifest_path = b / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["capture"]["target_saved_fps"] = 15
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="saved frame rates do not match"):
        import_pair(a, b, tmp_path / "out")

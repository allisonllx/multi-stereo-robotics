"""Config and YAML I/O must round-trip calibration matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibration.calibrate_intrinsics import IntrinsicsResult
from calibration.common import load_config, load_yaml, resolve_path, save_yaml


def test_default_config_has_pipeline_keys():
    config = load_config()
    assert set(config["cameras"]) == {"A", "B"}
    assert "square_length_m" in config["charuco"]
    assert "dictionary" in config["charuco"]
    assert config["sync"]["clap_window_s"] > 0


def test_intrinsics_yaml_roundtrip(tmp_path: Path):
    result = IntrinsicsResult(
        camera_matrix=np.array([[900.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]]),
        dist_coeffs=np.array([0.01, -0.02, 0.0, 0.0, 0.0]),
        reprojection_error_px=0.37,
        per_view_errors=np.array([0.2, 0.4]),
        image_width=1280,
        image_height=720,
        n_views_used=20,
        n_views_rejected=1,
    )
    path = tmp_path / "intrinsics_A.yaml"
    save_yaml(path, result.to_dict("A"))
    loaded = load_yaml(path)
    assert loaded["camera_id"] == "A"
    np.testing.assert_allclose(loaded["camera_matrix"], result.camera_matrix)
    assert loaded["reprojection_error_px"] == 0.37


def test_resolve_path_is_repo_relative():
    config = load_config()
    resolved = resolve_path(config, "data/camera_A.mp4")
    assert resolved.name == "camera_A.mp4"
    assert resolved.parent.name == "data"

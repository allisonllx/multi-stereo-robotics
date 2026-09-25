from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from processing.session import associate_nearest, load_frames, load_numeric_csv
from calibration.common import as_matrix, load_yaml
from calibration.pair_frames import load_pairs_csv

OPENCV_TO_ARKIT = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class PlanarAlignment:
    rotation: np.ndarray
    translation: np.ndarray
    rmse_m: float


@dataclass(frozen=True)
class PhoneWorldAlignment:
    b_world_to_a_world: np.ndarray
    translation_rmse_m: float
    rotation_rmse_deg: float


def camera_b_world_pose(world_from_a: np.ndarray, a_to_b: np.ndarray) -> np.ndarray:
    """Predict Camera B camera-to-world from Camera A and X_B=T_B_A X_A."""
    return np.asarray(world_from_a, dtype=np.float64) @ np.linalg.inv(np.asarray(a_to_b, dtype=np.float64))


def opencv_extrinsic_to_arkit(transform: np.ndarray) -> np.ndarray:
    """Convert a camera-to-camera transform from OpenCV to ARKit camera axes."""
    return OPENCV_TO_ARKIT @ np.asarray(transform, dtype=np.float64) @ OPENCV_TO_ARKIT


def world_from_rectified_camera(world_from_original: np.ndarray, rectification_cv: np.ndarray) -> np.ndarray:
    """Return the ARKit camera-to-world pose for an OpenCV-rectified image."""
    rectified_to_original_cv = np.eye(4)
    rectified_to_original_cv[:3, :3] = np.asarray(rectification_cv, dtype=np.float64).T
    rectified_to_original_arkit = opencv_extrinsic_to_arkit(rectified_to_original_cv)
    return np.asarray(world_from_original, dtype=np.float64) @ rectified_to_original_arkit


def align_phone_worlds(world_from_a_poses, world_from_b_poses, a_to_b: np.ndarray) -> PhoneWorldAlignment:
    if len(world_from_a_poses) != len(world_from_b_poses) or not world_from_a_poses:
        raise ValueError("paired A and B pose lists must have equal non-zero length")
    candidates = []
    for pose_a, pose_b in zip(world_from_a_poses, world_from_b_poses):
        expected_b_in_a_world = camera_b_world_pose(pose_a, a_to_b)
        candidates.append(expected_b_in_a_world @ np.linalg.inv(np.asarray(pose_b, dtype=np.float64)))
    rotations = Rotation.from_matrix(np.asarray([value[:3, :3] for value in candidates]))
    mean_rotation = rotations.mean()
    translation = np.mean([value[:3, 3] for value in candidates], axis=0)
    transform = np.eye(4); transform[:3, :3] = mean_rotation.as_matrix(); transform[:3, 3] = translation
    translation_errors, rotation_errors = [], []
    for candidate in candidates:
        delta = np.linalg.inv(transform) @ candidate
        translation_errors.append(np.linalg.norm(delta[:3, 3]))
        rotation_errors.append(Rotation.from_matrix(delta[:3, :3]).magnitude())
    return PhoneWorldAlignment(transform, float(np.sqrt(np.mean(np.square(translation_errors)))), float(np.rad2deg(np.sqrt(np.mean(np.square(rotation_errors))))))


def _geodetic_to_ecef(lat_deg, lon_deg, altitude_m):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    a, e2 = 6378137.0, 6.69437999014e-3
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (n + altitude_m) * np.cos(lat) * np.cos(lon)
    y = (n + altitude_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - e2) + altitude_m) * np.sin(lat)
    return np.column_stack([x, y, z])


def geodetic_to_enu(lat_deg, lon_deg, altitude_m, origin_lat_deg: float, origin_lon_deg: float, origin_altitude_m: float) -> np.ndarray:
    xyz = _geodetic_to_ecef(np.asarray(lat_deg), np.asarray(lon_deg), np.asarray(altitude_m))
    origin = _geodetic_to_ecef(np.asarray([origin_lat_deg]), np.asarray([origin_lon_deg]), np.asarray([origin_altitude_m]))[0]
    lat0, lon0 = np.deg2rad(origin_lat_deg), np.deg2rad(origin_lon_deg)
    rotation = np.array([[-np.sin(lon0), np.cos(lon0), 0], [-np.sin(lat0)*np.cos(lon0), -np.sin(lat0)*np.sin(lon0), np.cos(lat0)], [np.cos(lat0)*np.cos(lon0), np.cos(lat0)*np.sin(lon0), np.sin(lat0)]])
    return (rotation @ (xyz - origin).T).T


def fit_planar_rigid_alignment(source_xy: np.ndarray, target_xy: np.ndarray) -> PlanarAlignment:
    source, target = np.asarray(source_xy, dtype=np.float64), np.asarray(target_xy, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 2:
        raise ValueError("source and target must be matching Nx2 arrays with at least two points")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    U, _s, Vt = np.linalg.svd(covariance)
    rotation = Vt.T @ U.T
    if np.linalg.det(rotation) < 0:
        Vt[-1] *= -1; rotation = Vt.T @ U.T
    translation = target_center - rotation @ source_center
    residual = (rotation @ source.T).T + translation - target
    return PlanarAlignment(rotation, translation, float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))


def align_session_to_gps(session_dir: str | Path, output_csv: str | Path,
                         max_gps_age_s: float = 2.0, max_accuracy_m: float = 20.0) -> dict:
    session = Path(session_dir)
    frames = load_frames(session)
    names, locations = load_numeric_csv(session / "location.csv")
    column = {name: i for i, name in enumerate(names)}
    required = {"unix_time_s", "latitude", "longitude", "altitude_m", "horizontal_accuracy_m"}
    if not required <= set(column):
        raise ValueError(f"location.csv missing: {sorted(required - set(column))}")
    good = np.isfinite(locations[:, column["horizontal_accuracy_m"]]) & (locations[:, column["horizontal_accuracy_m"]] <= max_accuracy_m)
    locations = locations[good]
    if len(locations) < 2:
        raise ValueError("need at least two accurate GPS samples")
    frame_unix = np.array([float(f.values["unix_time_s"]) for f in frames])
    association = associate_nearest(frame_unix, locations[:, column["unix_time_s"]], max_gps_age_s)
    valid = association.indices >= 0
    selected = locations[association.indices[valid]]
    enu = geodetic_to_enu(selected[:, column["latitude"]], selected[:, column["longitude"]], selected[:, column["altitude_m"]], locations[0, column["latitude"]], locations[0, column["longitude"]], locations[0, column["altitude_m"]])
    local = np.array([[f.camera_to_world[0, 3], -f.camera_to_world[2, 3]] for f, keep in zip(frames, valid) if keep])
    fit = fit_planar_rigid_alignment(local, enu[:, :2])
    all_local = np.array([[f.camera_to_world[0, 3], -f.camera_to_world[2, 3]] for f in frames])
    aligned = (fit.rotation @ all_local.T).T + fit.translation
    output = Path(output_csv); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["frame_id", "timestamp_s", "east_m", "north_m", "up_m"])
        for frame, xy in zip(frames, aligned): writer.writerow([frame.frame_id, frame.timestamp_s, xy[0], xy[1], frame.camera_to_world[1, 3]])
    return {"frames": len(frames), "gps_matches": int(valid.sum()), "alignment_rmse_m": fit.rmse_m,
            "rotation_2x2": fit.rotation.tolist(), "translation_en_m": fit.translation.tolist()}


def align_pair_trajectories(session_a: str | Path, session_b: str | Path,
                            pairs_csv: str | Path, stereo_yaml: str | Path,
                            output_csv: str | Path) -> dict:
    frames_a, frames_b = load_frames(session_a), load_frames(session_b)
    pairs = load_pairs_csv(pairs_csv); stereo = load_yaml(stereo_yaml)
    a_to_b_cv = np.eye(4); a_to_b_cv[:3, :3] = as_matrix(stereo["R"]); a_to_b_cv[:3, 3] = as_matrix(stereo["T"]).reshape(3)
    a_to_b = opencv_extrinsic_to_arkit(a_to_b_cv)
    poses_a = [frames_a[p.frame_a].camera_to_world for p in pairs]
    poses_b = [frames_b[p.frame_b].camera_to_world for p in pairs]
    fit = align_phone_worlds(poses_a, poses_b, a_to_b)
    output = Path(output_csv); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["frame_id", "timestamp_s"] + [f"pose_m{r}{c}" for r in range(4) for c in range(4)])
        for frame in frames_b:
            pose = fit.b_world_to_a_world @ frame.camera_to_world
            writer.writerow([frame.frame_id, frame.timestamp_s] + pose.reshape(-1).tolist())
    report = {"paired_poses": len(pairs), "b_world_to_a_world": fit.b_world_to_a_world.tolist(), "translation_rmse_m": fit.translation_rmse_m, "rotation_rmse_deg": fit.rotation_rmse_deg}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

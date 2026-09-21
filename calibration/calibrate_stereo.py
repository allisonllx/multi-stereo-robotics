"""Stereo extrinsics with previously estimated intrinsics held fixed."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from calibration.common import (
    as_matrix,
    charuco_board_from_config,
    load_config,
    load_yaml,
    output_dir,
    require_cv2,
    save_yaml,
)
from calibration.detect_charuco import load_detections
from calibration.pair_frames import load_pairs_csv


@dataclass
class StereoResult:
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    rms_error: float
    camera_matrix_a: np.ndarray
    dist_a: np.ndarray
    camera_matrix_b: np.ndarray
    dist_b: np.ndarray
    image_width: int
    image_height: int
    n_pairs_used: int
    per_view_errors: np.ndarray | None = None

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.T))

    def to_dict(self) -> dict:
        return {
            "R": np.asarray(self.R, dtype=float).tolist(),
            "T": np.asarray(self.T, dtype=float).reshape(-1).tolist(),
            "E": np.asarray(self.E, dtype=float).tolist(),
            "F": np.asarray(self.F, dtype=float).tolist(),
            "baseline_m": self.baseline_m,
            "rms_reprojection_error_px": float(self.rms_error),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "n_pairs_used": int(self.n_pairs_used),
            "flags": "CALIB_FIX_INTRINSIC",
            "camera_matrix_A": np.asarray(self.camera_matrix_a, dtype=float).tolist(),
            "distortion_A": np.asarray(self.dist_a, dtype=float).ravel().tolist(),
            "camera_matrix_B": np.asarray(self.camera_matrix_b, dtype=float).tolist(),
            "distortion_B": np.asarray(self.dist_b, dtype=float).ravel().tolist(),
            "per_view_errors_px": None
            if self.per_view_errors is None
            else np.asarray(self.per_view_errors, dtype=float).ravel().tolist(),
        }


def intersect_view(
    object_points: np.ndarray,
    ids_a: np.ndarray,
    pts_a: np.ndarray,
    ids_b: np.ndarray,
    pts_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep ChArUco corner IDs observed in both cameras of one pair."""
    map_a = {int(i): pts_a[k] for k, i in enumerate(np.asarray(ids_a).reshape(-1))}
    map_b = {int(i): pts_b[k] for k, i in enumerate(np.asarray(ids_b).reshape(-1))}
    shared = sorted(set(map_a) & set(map_b))
    obj = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)[shared]
    ima = np.asarray([map_a[i] for i in shared], dtype=np.float32).reshape(-1, 2)
    imb = np.asarray([map_b[i] for i in shared], dtype=np.float32).reshape(-1, 2)
    return obj, ima, imb


def _index_detections(payload: dict) -> dict[int | str, dict]:
    indexed: dict[int | str, dict] = {}
    for record in payload["frames"]:
        frame = record["frame"]
        indexed[frame] = record
        if isinstance(frame, str) and frame.isdigit():
            indexed[int(frame)] = record
        elif isinstance(frame, int):
            indexed[str(frame)] = record
    return indexed


def paired_charuco_observations(
    pairs,
    detections_a: dict,
    detections_b: dict,
    board,
    min_corners: int,
    pose_interval_s: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict]]:
    object_points = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    index_a = _index_detections(detections_a)
    index_b = _index_detections(detections_b)

    objs: list[np.ndarray] = []
    imgs_a: list[np.ndarray] = []
    imgs_b: list[np.ndarray] = []
    meta: list[dict] = []
    last_kept_t: float | None = None

    for pair in pairs:
        rec_a = index_a.get(pair.frame_a) or index_a.get(str(pair.frame_a))
        rec_b = index_b.get(pair.frame_b) or index_b.get(str(pair.frame_b))
        if rec_a is None or rec_b is None:
            continue
        if pose_interval_s is not None and last_kept_t is not None:
            if abs(pair.timestamp_a - last_kept_t) < pose_interval_s:
                continue
        obj, ima, imb = intersect_view(
            object_points,
            np.asarray(rec_a["ids"]),
            np.asarray(rec_a["corners"], dtype=np.float32),
            np.asarray(rec_b["ids"]),
            np.asarray(rec_b["corners"], dtype=np.float32),
        )
        if obj.shape[0] < min_corners:
            continue
        objs.append(obj)
        imgs_a.append(ima)
        imgs_b.append(imb)
        meta.append(
            {
                "frame_A": pair.frame_a,
                "frame_B": pair.frame_b,
                "timestamp_A": pair.timestamp_a,
                "n_corners": int(obj.shape[0]),
            }
        )
        last_kept_t = pair.timestamp_a
    return objs, imgs_a, imgs_b, meta


def calibrate_stereo(
    object_points: list[np.ndarray],
    image_points_a: list[np.ndarray],
    image_points_b: list[np.ndarray],
    camera_matrix_a: np.ndarray,
    dist_a: np.ndarray,
    camera_matrix_b: np.ndarray,
    dist_b: np.ndarray,
    image_size: tuple[int, int],
) -> StereoResult:
    cv = require_cv2()
    if len(object_points) < 5:
        raise ValueError(f"Need at least 5 stereo poses, got {len(object_points)}")

    objs = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 3) for p in object_points]
    imgs_a = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in image_points_a]
    imgs_b = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in image_points_b]
    K_a = np.asarray(camera_matrix_a, dtype=np.float64).copy()
    K_b = np.asarray(camera_matrix_b, dtype=np.float64).copy()
    d_a = np.asarray(dist_a, dtype=np.float64).reshape(-1, 1).copy()
    d_b = np.asarray(dist_b, dtype=np.float64).reshape(-1, 1).copy()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    rms, K_a_out, d_a_out, K_b_out, d_b_out, R, T, E, F = cv.stereoCalibrate(
        objs,
        imgs_a,
        imgs_b,
        K_a,
        d_a,
        K_b,
        d_b,
        image_size,
        flags=cv.CALIB_FIX_INTRINSIC,
        criteria=criteria,
    )
    per_view = None

    return StereoResult(
        R=np.asarray(R),
        T=np.asarray(T).reshape(3, 1),
        E=np.asarray(E),
        F=np.asarray(F),
        rms_error=float(rms),
        camera_matrix_a=np.asarray(K_a_out),
        dist_a=np.asarray(d_a_out),
        camera_matrix_b=np.asarray(K_b_out),
        dist_b=np.asarray(d_b_out),
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
        n_pairs_used=len(object_points),
        per_view_errors=None if per_view is None else np.asarray(per_view).ravel(),
    )


def load_intrinsics(path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = load_yaml(path)
    K = as_matrix(data["camera_matrix"]).reshape(3, 3)
    dist = as_matrix(data["distortion_coefficients"]).reshape(-1)
    size = (int(data["image_width"]), int(data["image_height"]))
    return K, dist, size


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate stereo extrinsics with CALIB_FIX_INTRINSIC"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--detections-a", default=None)
    parser.add_argument("--detections-b", default=None)
    parser.add_argument("--intrinsics-a", default=None)
    parser.add_argument("--intrinsics-b", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    board = charuco_board_from_config(config)
    out_dir = output_dir(config)
    calib_cfg = config.get("calibration", {})

    pairs = load_pairs_csv(Path(args.pairs) if args.pairs else out_dir / "synchronization.csv")
    det_a = load_detections(Path(args.detections_a) if args.detections_a else out_dir / "detections_A.json")
    det_b = load_detections(Path(args.detections_b) if args.detections_b else out_dir / "detections_B.json")
    K_a, d_a, size_a = load_intrinsics(
        Path(args.intrinsics_a) if args.intrinsics_a else out_dir / "intrinsics_A.yaml"
    )
    K_b, d_b, size_b = load_intrinsics(
        Path(args.intrinsics_b) if args.intrinsics_b else out_dir / "intrinsics_B.yaml"
    )
    if size_a != size_b:
        print(f"warning: cameras have different resolutions {size_a} vs {size_b}")

    objs, imgs_a, imgs_b, meta = paired_charuco_observations(
        pairs,
        det_a,
        det_b,
        board,
        min_corners=int(calib_cfg.get("min_corners", 8)),
        pose_interval_s=float(calib_cfg.get("pose_interval_s", 0.8)),
    )
    result = calibrate_stereo(objs, imgs_a, imgs_b, K_a, d_a, K_b, d_b, size_a)
    payload = result.to_dict()
    payload["pairs"] = meta
    output = Path(args.output) if args.output else out_dir / "stereo_extrinsics.yaml"
    save_yaml(output, payload)
    print(
        f"stereo RMS {result.rms_error:.4f} px, baseline {result.baseline_m*1000:.1f} mm, "
        f"{result.n_pairs_used} poses -> {output}"
    )


if __name__ == "__main__":
    main()

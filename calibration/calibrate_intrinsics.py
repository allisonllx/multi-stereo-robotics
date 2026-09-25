"""Estimate per-camera intrinsics from calibration-board observations."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from calibration.common import (
    board_metadata,
    calibration_board_from_config,
    load_config,
    output_dir,
    require_cv2,
    save_yaml,
)
from calibration.detect_charuco import (
    detections_to_point_lists,
    load_detections,
    validate_detection_board,
)


@dataclass
class IntrinsicsResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    reprojection_error_px: float
    per_view_errors: np.ndarray
    image_width: int
    image_height: int
    n_views_used: int
    n_views_rejected: int
    flags: int = 0

    def to_dict(self, camera_id: str | None = None) -> dict:
        payload = {
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "camera_matrix": np.asarray(self.camera_matrix, dtype=float).tolist(),
            "distortion_coefficients": np.asarray(self.dist_coeffs, dtype=float).ravel().tolist(),
            "reprojection_error_px": float(self.reprojection_error_px),
            "per_view_errors_px": np.asarray(self.per_view_errors, dtype=float).ravel().tolist(),
            "n_views_used": int(self.n_views_used),
            "n_views_rejected": int(self.n_views_rejected),
        }
        if camera_id is not None:
            payload["camera_id"] = camera_id
        return payload


def _calibrate_once(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
):
    cv = require_cv2()
    objs = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 3) for p in object_points]
    imgs = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in image_points]
    (
        rms,
        K,
        dist,
        rvecs,
        tvecs,
        _std_int,
        _std_ext,
        per_view,
    ) = cv.calibrateCameraExtended(
        objs,
        imgs,
        image_size,
        None,
        None,
    )
    return float(rms), np.asarray(K), np.asarray(dist), rvecs, tvecs, np.asarray(per_view).ravel()


def calibrate_intrinsics(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    max_view_error_px: float = 2.0,
    min_views: int = 8,
) -> IntrinsicsResult:
    if len(object_points) < 4:
        raise ValueError(f"Need at least 4 calibration-board views, got {len(object_points)}")

    rms, K, dist, _rvecs, _tvecs, per_view = _calibrate_once(
        object_points, image_points, image_size
    )
    keep = per_view < max_view_error_px
    n_rejected = int(np.size(per_view) - np.count_nonzero(keep))
    if n_rejected and int(np.count_nonzero(keep)) >= min_views:
        kept_obj = [p for p, ok in zip(object_points, keep) if ok]
        kept_img = [p for p, ok in zip(image_points, keep) if ok]
        rms, K, dist, _rvecs, _tvecs, per_view = _calibrate_once(
            kept_obj, kept_img, image_size
        )
        n_used = len(kept_obj)
    else:
        n_used = len(object_points)
        n_rejected = 0 if int(np.count_nonzero(keep)) < min_views else n_rejected

    return IntrinsicsResult(
        camera_matrix=K,
        dist_coeffs=dist,
        reprojection_error_px=rms,
        per_view_errors=per_view,
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
        n_views_used=n_used,
        n_views_rejected=n_rejected,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate one camera's intrinsics from board detections")
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", choices=("A", "B"), default=None)
    parser.add_argument("--detections", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    board = calibration_board_from_config(config)
    out_dir = output_dir(config)
    calib_cfg = config.get("calibration", {})
    cameras = ("A", "B") if args.camera is None else (args.camera,)

    for camera in cameras:
        det_path = Path(args.detections) if args.detections else None
        if det_path is None:
            dedicated = out_dir / f"detections_intrinsics_{camera}.json"
            det_path = dedicated if dedicated.exists() else out_dir / f"detections_{camera}.json"
        payload = load_detections(det_path)
        validate_detection_board(payload, board)
        objs, imgs, _meta = detections_to_point_lists(payload, board)
        result = calibrate_intrinsics(
            objs,
            imgs,
            (int(payload["image_width"]), int(payload["image_height"])),
            max_view_error_px=float(calib_cfg.get("max_view_error_px", 2.0)),
            min_views=int(calib_cfg.get("min_views", 8)),
        )
        output = Path(args.output) if args.output else out_dir / f"intrinsics_{camera}.yaml"
        result_payload = result.to_dict(camera_id=camera)
        result_payload["board"] = board_metadata(board)
        save_yaml(output, result_payload)
        print(
            f"{camera}: RMS {result.reprojection_error_px:.4f} px, "
            f"{result.n_views_used} views used, {result.n_views_rejected} rejected -> {output}"
        )
        if result.reprojection_error_px > 0.5:
            print("  warning: RMS is above the ~0.5 px diagnostic target")


if __name__ == "__main__":
    main()

"""Rectify the stereo rig and report held-out geometric diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from calibration.calibrate_stereo import load_intrinsics, paired_charuco_observations
from calibration.common import (
    as_matrix,
    board_metadata,
    calibration_board_from_config,
    load_config,
    load_yaml,
    output_dir,
    require_cv2,
    save_yaml,
)
from calibration.detect_charuco import load_detections, validate_detection_board
from calibration.pair_frames import load_pairs_csv


def epipolar_distances(pts_a: np.ndarray, pts_b: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Symmetric point-to-epipolar-line distance in pixels."""
    a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    F = np.asarray(F, dtype=np.float64).reshape(3, 3)
    ones = np.ones((a.shape[0], 1))
    x1 = np.hstack([a, ones])
    x2 = np.hstack([b, ones])
    lines_2 = (F @ x1.T).T
    lines_1 = (F.T @ x2.T).T
    num_2 = np.abs(np.sum(x2 * lines_2, axis=1))
    num_1 = np.abs(np.sum(x1 * lines_1, axis=1))
    den_2 = np.sqrt(lines_2[:, 0] ** 2 + lines_2[:, 1] ** 2) + 1e-15
    den_1 = np.sqrt(lines_1[:, 0] ** 2 + lines_1[:, 1] ** 2) + 1e-15
    return 0.5 * (num_1 / den_1 + num_2 / den_2)


def mean_vertical_disparity(pts_a_rect: np.ndarray, pts_b_rect: np.ndarray) -> float:
    a = np.asarray(pts_a_rect, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(pts_b_rect, dtype=np.float64).reshape(-1, 2)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a[:, 1] - b[:, 1])))


def adjacent_charuco_pairs(squares_x: int, squares_y: int) -> list[tuple[int, int]]:
    nx, ny = squares_x - 1, squares_y - 1
    pairs: list[tuple[int, int]] = []
    for y in range(ny):
        for x in range(nx):
            index = y * nx + x
            if x + 1 < nx:
                pairs.append((index, index + 1))
            if y + 1 < ny:
                pairs.append((index, index + nx))
    return pairs


def triangulate_points(
    K_a: np.ndarray,
    dist_a: np.ndarray,
    K_b: np.ndarray,
    dist_b: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
) -> np.ndarray:
    cv = require_cv2()
    pts_a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 1, 2)
    pts_b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 1, 2)
    und_a = cv.undistortPoints(pts_a, K_a, dist_a)
    und_b = cv.undistortPoints(pts_b, K_b, dist_b)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([np.asarray(R, dtype=np.float64), np.asarray(T, dtype=np.float64).reshape(3, 1)])
    homog = cv.triangulatePoints(P1, P2, und_a.reshape(-1, 2).T, und_b.reshape(-1, 2).T)
    homog = homog / homog[3]
    return homog[:3].T


def triangulated_square_lengths(
    xyz: np.ndarray,
    ids: np.ndarray,
    squares_x: int,
    squares_y: int,
) -> np.ndarray:
    id_to_xyz = {int(i): xyz[k] for k, i in enumerate(np.asarray(ids).reshape(-1))}
    lengths = []
    for left, right in adjacent_charuco_pairs(squares_x, squares_y):
        if left in id_to_xyz and right in id_to_xyz:
            lengths.append(float(np.linalg.norm(id_to_xyz[left] - id_to_xyz[right])))
    return np.asarray(lengths, dtype=np.float64)


def stereo_rectify_maps(intrinsics_a, dist_a, intrinsics_b, dist_b, image_size, R, T, alpha: float = 0):
    cv = require_cv2()
    T = np.asarray(T, dtype=np.float64).reshape(3, 1)
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
        intrinsics_a,
        dist_a,
        intrinsics_b,
        dist_b,
        image_size,
        R,
        T,
        flags=cv.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )
    map1_a, map2_a = cv.initUndistortRectifyMap(intrinsics_a, dist_a, R1, P1, image_size, cv.CV_32FC1)
    map1_b, map2_b = cv.initUndistortRectifyMap(intrinsics_b, dist_b, R2, P2, image_size, cv.CV_32FC1)
    return {
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "roi1": roi1,
        "roi2": roi2,
        "map1_a": map1_a,
        "map2_a": map2_a,
        "map1_b": map1_b,
        "map2_b": map2_b,
    }


def _rectify_points(pts, K, dist, R_rect, P):
    cv = require_cv2()
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    und = cv.undistortPoints(pts, K, dist, R=R_rect, P=P)
    return und.reshape(-1, 2)


def _split_holdout(items: list, holdout_fraction: float) -> tuple[list, list]:
    if not items:
        return [], []
    n_hold = max(1, int(round(len(items) * holdout_fraction))) if len(items) > 4 else 0
    if n_hold == 0:
        return items, []
    return items[:-n_hold], items[-n_hold:]


def verify(
    objs: list[np.ndarray],
    imgs_a: list[np.ndarray],
    imgs_b: list[np.ndarray],
    meta: list[dict],
    stereo: dict,
    board_cfg: dict,
    holdout_fraction: float,
    measured_baseline_m: float | None,
    intrinsic_error_a: float | None = None,
    intrinsic_error_b: float | None = None,
    sync_residuals_ms: np.ndarray | None = None,
) -> dict:
    K_a = as_matrix(stereo["camera_matrix_A"]).reshape(3, 3)
    K_b = as_matrix(stereo["camera_matrix_B"]).reshape(3, 3)
    d_a = as_matrix(stereo["distortion_A"])
    d_b = as_matrix(stereo["distortion_B"])
    R = as_matrix(stereo["R"]).reshape(3, 3)
    T = as_matrix(stereo["T"]).reshape(3, 1)
    F = as_matrix(stereo["F"]).reshape(3, 3)
    image_size = (int(stereo["image_width"]), int(stereo["image_height"]))

    train, hold = _split_holdout(list(range(len(objs))), holdout_fraction)
    rect = stereo_rectify_maps(K_a, d_a, K_b, d_b, image_size, R, T)

    epi = []
    vert = []
    square_lengths = []
    eval_indices = hold if hold else train
    square = float(board_cfg["square_length_m"])
    for i in eval_indices:
        epi.append(epipolar_distances(imgs_a[i], imgs_b[i], F))
        pts_a_r = _rectify_points(imgs_a[i], K_a, d_a, rect["R1"], rect["P1"])
        pts_b_r = _rectify_points(imgs_b[i], K_b, d_b, rect["R2"], rect["P2"])
        vert.append(mean_vertical_disparity(pts_a_r, pts_b_r))
        xyz = triangulate_points(K_a, d_a, K_b, d_b, R, T, imgs_a[i], imgs_b[i])
        obj = np.asarray(objs[i]).reshape(-1, 3)
        for p, q in _adjacent_object_pairs(obj, square):
            square_lengths.append(float(np.linalg.norm(xyz[p] - xyz[q])))

    square_lengths = np.asarray(square_lengths, dtype=np.float64)
    epi_all = np.concatenate(epi) if epi else np.zeros(0)
    baseline = float(np.linalg.norm(T))
    report = {
        "n_train_poses": len(train),
        "n_holdout_poses": len(hold),
        "mean_intrinsic_reprojection_error_px": {
            "A": intrinsic_error_a,
            "B": intrinsic_error_b,
        },
        "mean_epipolar_distance_px": None if epi_all.size == 0 else float(np.mean(epi_all)),
        "median_epipolar_distance_px": None if epi_all.size == 0 else float(np.median(epi_all)),
        "mean_vertical_disparity_px": None if not vert else float(np.mean(vert)),
        "baseline_m": baseline,
        "measured_baseline_m": measured_baseline_m,
        "baseline_error_m": None
        if measured_baseline_m is None
        else abs(baseline - measured_baseline_m),
        "triangulated_square_length_mean_m": None
        if square_lengths.size == 0
        else float(np.mean(square_lengths)),
        "triangulated_square_length_std_m": None
        if square_lengths.size == 0
        else float(np.std(square_lengths)),
        "known_square_length_m": float(board_cfg["square_length_m"]),
        "sync_residual_mean_abs_ms": None
        if sync_residuals_ms is None or sync_residuals_ms.size == 0
        else float(np.mean(np.abs(sync_residuals_ms))),
        "sync_residual_max_abs_ms": None
        if sync_residuals_ms is None or sync_residuals_ms.size == 0
        else float(np.max(np.abs(sync_residuals_ms))),
        "targets": {
            "intrinsic_reprojection_error_px": 0.5,
            "epipolar_or_vertical_error_px": 1.0,
            "sync_residual_frames": 1.0,
        },
        "rectification": {
            "R1": np.asarray(rect["R1"]).tolist(),
            "R2": np.asarray(rect["R2"]).tolist(),
            "P1": np.asarray(rect["P1"]).tolist(),
            "P2": np.asarray(rect["P2"]).tolist(),
            "Q": np.asarray(rect["Q"]).tolist(),
            "roi1": list(rect["roi1"]),
            "roi2": list(rect["roi2"]),
        },
    }
    return report


def _adjacent_object_pairs(obj: np.ndarray, square_length: float, tol: float = 1e-4) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(obj)):
        for j in range(i + 1, len(obj)):
            if abs(float(np.linalg.norm(obj[i] - obj[j])) - square_length) < tol:
                pairs.append((i, j))
    return pairs


def _maybe_write_preview(
    config: dict,
    rect: dict,
    pairs,
    out_dir: Path,
) -> str | None:
    cv = require_cv2()
    cameras = config.get("cameras", {})
    video_a = cameras.get("A", {}).get("video")
    video_b = cameras.get("B", {}).get("video")
    if not video_a or not video_b or not pairs:
        return None
    from calibration.common import resolve_path

    path_a = resolve_path(config, video_a)
    path_b = resolve_path(config, video_b)
    if not path_a.exists() or not path_b.exists():
        return None

    pair = pairs[len(pairs) // 2]
    cap_a = cv.VideoCapture(str(path_a))
    cap_b = cv.VideoCapture(str(path_b))
    cap_a.set(cv.CAP_PROP_POS_FRAMES, pair.frame_a)
    cap_b.set(cv.CAP_PROP_POS_FRAMES, pair.frame_b)
    ok_a, frame_a = cap_a.read()
    ok_b, frame_b = cap_b.read()
    cap_a.release()
    cap_b.release()
    if not ok_a or not ok_b:
        return None
    rect_a = cv.remap(frame_a, rect["map1_a"], rect["map2_a"], cv.INTER_LINEAR)
    rect_b = cv.remap(frame_b, rect["map1_b"], rect["map2_b"], cv.INTER_LINEAR)
    height = min(rect_a.shape[0], rect_b.shape[0])
    width = min(rect_a.shape[1], rect_b.shape[1])
    canvas = np.hstack([rect_a[:height, :width], rect_b[:height, :width]])
    for y in range(40, height, 80):
        cv.line(canvas, (0, y), (canvas.shape[1] - 1, y), (0, 255, 0), 1)
    preview = out_dir / "rectified_preview.png"
    cv.imwrite(str(preview), canvas)
    return str(preview)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify stereo calibration on held-out poses")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    board = calibration_board_from_config(config)
    board_type = str(config.get("board", {}).get("type", "charuco")).lower()
    board_cfg = config[board_type]
    out_dir = output_dir(config)
    calib_cfg = config.get("calibration", {})
    verify_cfg = config.get("verification", {})

    stereo = load_yaml(out_dir / "stereo_extrinsics.yaml")
    det_a = load_detections(out_dir / "detections_A.json")
    det_b = load_detections(out_dir / "detections_B.json")
    validate_detection_board(det_a, board)
    validate_detection_board(det_b, board)
    pairs = load_pairs_csv(out_dir / "synchronization.csv")
    objs, imgs_a, imgs_b, meta = paired_charuco_observations(
        pairs,
        det_a,
        det_b,
        board,
        min_corners=int(calib_cfg.get("min_corners", 8)),
        pose_interval_s=float(calib_cfg.get("pose_interval_s", 0.8)),
    )
    try:
        err_a = float(load_yaml(out_dir / "intrinsics_A.yaml")["reprojection_error_px"])
        err_b = float(load_yaml(out_dir / "intrinsics_B.yaml")["reprojection_error_px"])
    except (FileNotFoundError, KeyError):
        err_a = err_b = None

    sync_residuals = np.array([p.residual_s * 1000.0 for p in pairs], dtype=np.float64)
    report = verify(
        objs,
        imgs_a,
        imgs_b,
        meta,
        stereo,
        board_cfg,
        holdout_fraction=float(verify_cfg.get("holdout_fraction", 0.2)),
        measured_baseline_m=verify_cfg.get("measured_baseline_m"),
        intrinsic_error_a=err_a,
        intrinsic_error_b=err_b,
        sync_residuals_ms=sync_residuals,
    )
    report["board"] = board_metadata(board)

    K_a, d_a, size_a = load_intrinsics(out_dir / "intrinsics_A.yaml")
    K_b, d_b, _size_b = load_intrinsics(out_dir / "intrinsics_B.yaml")
    rect = stereo_rectify_maps(
        K_a,
        d_a,
        K_b,
        d_b,
        size_a,
        as_matrix(stereo["R"]),
        as_matrix(stereo["T"]),
    )
    preview = _maybe_write_preview(config, rect, pairs, out_dir)
    if preview:
        report["rectified_preview"] = preview

    output = Path(args.output) if args.output else out_dir / "verification_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_yaml(out_dir / "rectification.yaml", report["rectification"])
    print(f"Wrote {output}")
    print(f"  epipolar {report['mean_epipolar_distance_px']} px")
    print(f"  vertical disparity {report['mean_vertical_disparity_px']} px")
    print(f"  baseline {report['baseline_m']} m")
    print(f"  triangulated square {report['triangulated_square_length_mean_m']} m")


if __name__ == "__main__":
    main()

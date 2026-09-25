from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration.common import as_matrix, load_yaml, require_cv2
from calibration.pair_frames import load_pairs_csv
from calibration.verify_calibration import stereo_rectify_maps
from processing.session import decode_padded_buffer, load_frames


def disparity_to_depth(disparity: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    disparity = np.asarray(disparity, dtype=np.float32)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    depth[valid] = float(focal_px) * float(baseline_m) / disparity[valid]
    return depth


def evaluate_depth(estimate: np.ndarray, reference: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have the same shape")
    valid = np.isfinite(estimate) & np.isfinite(reference) & (estimate > 0) & (reference > 0)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    error = estimate[valid] - reference[valid]
    if error.size == 0:
        return {"valid_pixels": 0, "mae_m": None, "rmse_m": None, "median_abs_error_m": None, "mean_relative_error": None}
    return {
        "valid_pixels": int(error.size),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "median_abs_error_m": float(np.median(np.abs(error))),
        "mean_relative_error": float(np.mean(np.abs(error) / reference[valid])),
    }


def rectify_aligned_map(source: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
                        rgb_size: tuple[int, int], interpolation: int | None = None) -> np.ndarray:
    """Warp an RGB-aligned lower-resolution map into the rectified RGB plane."""
    cv = require_cv2()
    source = np.asarray(source)
    rgb_width, rgb_height = rgb_size
    scaled_x = np.asarray(map_x, np.float32) * (source.shape[1] / float(rgb_width))
    scaled_y = np.asarray(map_y, np.float32) * (source.shape[0] / float(rgb_height))
    mode = cv.INTER_NEAREST if interpolation is None else interpolation
    return cv.remap(source, scaled_x, scaled_y, mode, borderMode=cv.BORDER_CONSTANT, borderValue=0)


def compute_disparity(left: np.ndarray, right: np.ndarray, num_disparities: int = 128, block_size: int = 5) -> np.ndarray:
    cv = require_cv2()
    if num_disparities <= 0 or num_disparities % 16:
        raise ValueError("num_disparities must be a positive multiple of 16")
    gray_l = cv.cvtColor(left, cv.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gray_r = cv.cvtColor(right, cv.COLOR_BGR2GRAY) if right.ndim == 3 else right
    matcher = cv.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
        P1=8 * block_size**2, P2=32 * block_size**2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=2, preFilterCap=31, mode=cv.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan
    return disparity


def process_sequence(session_a: str | Path, session_b: str | Path, pairs_csv: str | Path,
                     stereo_yaml: str | Path, output_dir: str | Path,
                     max_pairs: int | None = None) -> dict:
    cv = require_cv2()
    frames_a, frames_b = load_frames(session_a), load_frames(session_b)
    pairs = load_pairs_csv(pairs_csv)
    stereo = load_yaml(stereo_yaml)
    size = (int(stereo["image_width"]), int(stereo["image_height"]))
    K_a, K_b = as_matrix(stereo["camera_matrix_A"]), as_matrix(stereo["camera_matrix_B"])
    d_a, d_b = as_matrix(stereo["distortion_A"]), as_matrix(stereo["distortion_B"])
    R, T = as_matrix(stereo["R"]), as_matrix(stereo["T"])
    rect = stereo_rectify_maps(K_a, d_a, K_b, d_b, size, R, T)
    focal, baseline = float(rect["P1"][0, 0]), float(np.linalg.norm(T))
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(exist_ok=True); (output / "disparity").mkdir(exist_ok=True); (output / "rgb").mkdir(exist_ok=True)
    frame_metrics = []
    selected = pairs if max_pairs is None else pairs[:max_pairs]
    for pair in selected:
        fa, fb = frames_a[pair.frame_a], frames_b[pair.frame_b]
        left, right = cv.imread(str(fa.image_path)), cv.imread(str(fb.image_path))
        if left is None or right is None:
            raise RuntimeError(f"failed to read RGB pair {pair.frame_a}/{pair.frame_b}")
        left_r = cv.remap(left, rect["map1_a"], rect["map2_a"], cv.INTER_LINEAR)
        right_r = cv.remap(right, rect["map1_b"], rect["map2_b"], cv.INTER_LINEAR)
        disparity = compute_disparity(left_r, right_r)
        depth = disparity_to_depth(disparity, focal, baseline)
        stem = f"{pair.frame_a:06d}"
        np.save(output / "depth" / f"{stem}.npy", depth)
        np.save(output / "disparity" / f"{stem}.npy", disparity)
        if not cv.imwrite(str(output / "rgb" / f"{stem}.jpg"), left_r):
            raise RuntimeError(f"failed to write rectified RGB frame {stem}")
        metric = {"frame_A": pair.frame_a, "frame_B": pair.frame_b, "timestamp_s": pair.timestamp_a}
        row = fa.values
        if row.get("depth_path"):
            lidar = decode_padded_buffer(Path(session_a) / row["depth_path"], int(row["depth_width"]), int(row["depth_height"]), int(row["depth_bytes_per_row"]), np.float32)
            lidar_rect = rectify_aligned_map(lidar, rect["map1_a"], rect["map2_a"], size)
            confidence = None
            if row.get("confidence_path"):
                confidence_raw = decode_padded_buffer(Path(session_a) / row["confidence_path"], int(row["confidence_width"]), int(row["confidence_height"]), int(row["confidence_bytes_per_row"]), np.uint8)
                confidence = rectify_aligned_map(confidence_raw, rect["map1_a"], rect["map2_a"], size) == 2
            metric.update(evaluate_depth(depth, lidar_rect, confidence))
        frame_metrics.append(metric)
    valid = [m for m in frame_metrics if m.get("mae_m") is not None]
    report = {
        "pairs_processed": len(frame_metrics), "lidar_pairs_evaluated": len(valid),
        "mean_mae_m": None if not valid else float(np.mean([m["mae_m"] for m in valid])),
        "mean_rmse_m": None if not valid else float(np.mean([m["rmse_m"] for m in valid])),
        "frames": frame_metrics,
    }
    (output / "depth_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    metadata = {"image_width": size[0], "image_height": size[1], "rectified_intrinsics": np.asarray(rect["P1"][:3, :3]).tolist(), "rectification_R1": np.asarray(rect["R1"]).tolist(), "frames": frame_metrics}
    (output / "stereo_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return report

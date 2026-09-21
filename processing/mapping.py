from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration.common import require_cv2
from processing.session import load_frames


def backproject_depth(depth_m: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray,
                      pixel_stride: int = 1, max_depth_m: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float64); K = np.asarray(intrinsics, dtype=np.float64)
    v, u = np.mgrid[0:depth.shape[0]:pixel_stride, 0:depth.shape[1]:pixel_stride]
    z = depth[::pixel_stride, ::pixel_stride]
    valid = np.isfinite(z) & (z > 0) & (z <= max_depth_m)
    u, v, z = u[valid], v[valid], z[valid]
    x = (u - K[0, 2]) * z / K[0, 0]; y = (v - K[1, 2]) * z / K[1, 1]
    # OpenCV camera (right, down, forward) -> ARKit camera (right, up, backward).
    camera_points = np.column_stack([x, -y, -z, np.ones_like(z)])
    world = (np.asarray(camera_to_world, dtype=np.float64) @ camera_points.T).T[:, :3]
    return world, np.column_stack([v, u]).astype(int)


def voxel_downsample(points: np.ndarray, colours: np.ndarray, voxel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    points, colours = np.asarray(points), np.asarray(colours)
    if voxel_size_m <= 0: raise ValueError("voxel_size_m must be positive")
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    result_points = np.column_stack([np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)])
    result_colours = np.column_stack([np.bincount(inverse, weights=colours[:, axis]) / counts for axis in range(3)]).round().astype(np.uint8)
    return result_points, result_colours


def write_ply(path: str | Path, points: np.ndarray, colours: np.ndarray) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    vertices = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vertices["x"], vertices["y"], vertices["z"] = np.asarray(points, np.float32).T
    vertices["r"], vertices["g"], vertices["b"] = np.asarray(colours, np.uint8).T
    with path.open("wb") as handle: handle.write(header.encode("ascii")); handle.write(vertices.tobytes())
    return path


def build_coloured_map(session_dir: str | Path, processed_dir: str | Path, output_ply: str | Path,
                       frame_stride: int = 5, pixel_stride: int = 4,
                       voxel_size_m: float = 0.05, max_depth_m: float = 20.0) -> dict:
    cv = require_cv2(); session, processed = Path(session_dir), Path(processed_dir)
    frames = load_frames(session); metadata = json.loads((processed / "stereo_metadata.json").read_text(encoding="utf-8"))
    K = np.asarray(metadata["rectified_intrinsics"], dtype=np.float64)
    all_points, all_colours, used = [], [], 0
    for frame in frames[::frame_stride]:
        depth_path, image_path = processed / "depth" / f"{frame.frame_id:06d}.npy", processed / "rgb" / f"{frame.frame_id:06d}.jpg"
        if not depth_path.is_file() or not image_path.is_file(): continue
        depth, image = np.load(depth_path), cv.imread(str(image_path))
        points, pixels = backproject_depth(depth, K, frame.camera_to_world, pixel_stride, max_depth_m)
        colours = image[pixels[:, 0], pixels[:, 1], ::-1]
        all_points.append(points); all_colours.append(colours); used += 1
    if not all_points: raise ValueError("no matching rectified RGB/depth frames found")
    points, colours = voxel_downsample(np.vstack(all_points), np.vstack(all_colours), voxel_size_m)
    write_ply(output_ply, points, colours)
    return {"frames_used": used, "points": len(points), "voxel_size_m": voxel_size_m, "output": str(output_ply)}

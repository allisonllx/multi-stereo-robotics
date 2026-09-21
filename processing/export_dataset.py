from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.common import require_cv2


@dataclass(frozen=True)
class DatasetFrame:
    timestamp_s: float
    image_path: Path
    depth_m: np.ndarray
    confidence: np.ndarray
    camera_to_world: np.ndarray
    intrinsics: np.ndarray


def _pose_line(timestamp: float, pose: np.ndarray) -> str:
    pose = np.asarray(pose, dtype=np.float64)
    qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
    tx, ty, tz = pose[:3, 3]
    return f"{timestamp:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"


def export_tum(frames: Sequence[DatasetFrame], output_dir: str | Path) -> Path:
    if not frames: raise ValueError("cannot export an empty dataset")
    cv = require_cv2(); output = Path(output_dir)
    rgb_dir, depth_dir = output / "rgb", output / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True); depth_dir.mkdir(parents=True, exist_ok=True)
    color, depths, groundtruth, odom = [], [], [], []
    for index, frame in enumerate(frames):
        rgb_rel, depth_rel = f"rgb/{index:06d}.jpg", f"depth/{index:06d}.png"
        shutil.copy2(frame.image_path, output / rgb_rel)
        millimetres = np.nan_to_num(frame.depth_m, nan=0, posinf=0, neginf=0)
        millimetres = np.clip(np.rint(millimetres * 1000), 0, 65535).astype(np.uint16)
        if not cv.imwrite(str(output / depth_rel), millimetres): raise RuntimeError("failed to write depth PNG")
        color.append(f"{frame.timestamp_s:.9f} {rgb_rel}")
        depths.append(f"{frame.timestamp_s:.9f} {depth_rel}")
        pose = _pose_line(frame.timestamp_s, frame.camera_to_world)
        groundtruth.append(pose); odom.append(pose + " 0 0 0 0 0 0")
    (output / "color.txt").write_text("\n".join(color) + "\n", encoding="utf-8")
    (output / "aligned_depth.txt").write_text("\n".join(depths) + "\n", encoding="utf-8")
    (output / "groundtruth.txt").write_text("\n".join(groundtruth) + "\n", encoding="utf-8")
    (output / "odom.txt").write_text("\n".join(odom) + "\n", encoding="utf-8")
    h, w = np.asarray(frames[0].depth_m).shape
    K = np.asarray(frames[0].intrinsics); fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    sensors = f"""%YAML:1.0
d400_color_optical_frame:
  width: {w}
  height: {h}
  intrinsics: !!opencv-matrix
    rows: 1
    cols: 4
    dt: d
    data: [{fx}, {fy}, {cx}, {cy}]
d400_depth_optical_frame:
  width: {w}
  height: {h}
  intrinsics: !!opencv-matrix
    rows: 1
    cols: 4
    dt: d
    data: [{fx}, {fy}, {cx}, {cy}]
"""
    (output / "sensors.yaml").write_text(sensors, encoding="utf-8")
    transform = """%YAML:1.0
trans_matrix:
  - { child_frame: d400_color_optical_frame, parent_frame: base_link, matrix: !!opencv-matrix { rows: 4, cols: 4, dt: d, data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] } }
  - { child_frame: d400_depth_optical_frame, parent_frame: d400_color_optical_frame, matrix: !!opencv-matrix { rows: 4, cols: 4, dt: d, data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] } }
"""
    (output / "trans_matrix.yaml").write_text(transform, encoding="utf-8")
    return output


def _r3d_pose(pose: np.ndarray) -> list[float]:
    q = Rotation.from_matrix(np.asarray(pose)[:3, :3]).as_quat().tolist()
    return q + np.asarray(pose)[:3, 3].astype(float).tolist()


def export_r3d(frames: Sequence[DatasetFrame], archive_path: str | Path,
               compressor: Callable[[bytes], bytes] | None = None) -> Path:
    if not frames: raise ValueError("cannot export an empty dataset")
    if compressor is None:
        try:
            import lzfse
        except ImportError as error:
            raise RuntimeError("R3D export requires `pip install lzfse`") from error
        compressor = lzfse.compress
    cv = require_cv2(); archive = Path(archive_path); archive.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([f.timestamp_s for f in frames]); fps = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 1.0
    image = cv.imread(str(frames[0].image_path)); h, w = image.shape[:2]
    K = np.asarray(frames[0].intrinsics, dtype=float)
    poses = [_r3d_pose(frame.camera_to_world) for frame in frames]
    metadata = {"w": w, "h": h, "dw": 192, "dh": 256, "fps": float(fps), "K": K.T.reshape(-1).tolist(), "poses": poses, "initPose": poses[0]}
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zipped:
        zipped.writestr("metadata", json.dumps(metadata))
        for index, frame in enumerate(frames):
            zipped.write(frame.image_path, f"rgbd/{index}.jpg")
            depth = cv.resize(np.asarray(frame.depth_m, np.float32), (192, 256), interpolation=cv.INTER_NEAREST)
            conf = cv.resize(np.asarray(frame.confidence, np.uint8), (192, 256), interpolation=cv.INTER_NEAREST)
            zipped.writestr(f"rgbd/{index}.depth", compressor(depth.tobytes()))
            zipped.writestr(f"rgbd/{index}.conf", compressor(conf.tobytes()))
    return archive

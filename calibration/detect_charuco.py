"""Detect calibration-board corners and dump per-frame observations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from calibration.common import (
    ChessboardSpec,
    board_metadata,
    calibration_board_from_config,
    charuco_detector,
    load_config,
    output_dir,
    require_cv2,
    resolve_path,
)


@dataclass
class CharucoDetection:
    corners: np.ndarray
    ids: np.ndarray

    @property
    def n_corners(self) -> int:
        return 0 if self.ids is None else int(len(self.ids))


def refine_corners_subpixel(
    image: np.ndarray,
    corners: np.ndarray,
    win_size: tuple[int, int] = (5, 5),
) -> np.ndarray:
    cv = require_cv2()
    pts = np.asarray(corners, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape(0, 2)
    gray = image if image.ndim == 2 else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    reshaped = pts.reshape(-1, 1, 2)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv.cornerSubPix(gray, reshaped, win_size, (-1, -1), criteria)
    return refined.reshape(-1, 2).astype(np.float32)


def detect_charuco_image(
    image: np.ndarray,
    board,
    refine: bool = True,
) -> CharucoDetection:
    detector = charuco_detector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)
    if charuco_ids is None or len(charuco_ids) == 0:
        return CharucoDetection(corners=np.zeros((0, 2), dtype=np.float32), ids=np.zeros((0,), dtype=np.int32))

    corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
    ids = np.asarray(charuco_ids).reshape(-1).astype(np.int32)
    if refine:
        corners = refine_corners_subpixel(image, corners)
    return CharucoDetection(corners=corners, ids=ids)


def detect_chessboard_image(
    image: np.ndarray,
    board: ChessboardSpec,
    refine: bool = True,
) -> CharucoDetection:
    cv = require_cv2()
    gray = image if image.ndim == 2 else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    flags = cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
    found, corners = cv.findChessboardCornersSB(gray, board.pattern_size, flags=flags)
    if not found or corners is None:
        return CharucoDetection(
            corners=np.zeros((0, 2), dtype=np.float32), ids=np.zeros((0,), dtype=np.int32)
        )
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if refine:
        points = refine_corners_subpixel(gray, points)
    return CharucoDetection(corners=points, ids=np.arange(len(points), dtype=np.int32))


def detect_calibration_image(image: np.ndarray, board, refine: bool = True) -> CharucoDetection:
    if isinstance(board, ChessboardSpec):
        return detect_chessboard_image(image, board, refine=refine)
    return detect_charuco_image(image, board, refine=refine)


def common_corner_ids(
    ids_a: np.ndarray,
    corners_a: np.ndarray,
    ids_b: np.ndarray,
    corners_b: np.ndarray,
    object_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shared IDs and the corresponding 2D corners, ordered by ID."""
    map_a = {int(i): corners_a[k] for k, i in enumerate(np.asarray(ids_a).reshape(-1))}
    map_b = {int(i): corners_b[k] for k, i in enumerate(np.asarray(ids_b).reshape(-1))}
    shared = sorted(set(map_a) & set(map_b))
    ids = np.asarray(shared, dtype=np.int32)
    pts_a = np.asarray([map_a[i] for i in shared], dtype=np.float32).reshape(-1, 2)
    pts_b = np.asarray([map_b[i] for i in shared], dtype=np.float32).reshape(-1, 2)
    if object_points is None:
        return ids, pts_a, pts_b
    return ids, pts_a, pts_b


def _iter_images_from_dir(directory: Path):
    cv = require_cv2()
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in suffixes:
            continue
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            continue
        frame_id = int(path.stem) if path.stem.isdigit() else path.name
        yield frame_id, None, image


def _iter_frames_from_video(video_path: Path, stride: int):
    cv = require_cv2()
    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                timestamp_s = float(capture.get(cv.CAP_PROP_POS_MSEC)) / 1000.0
                yield index, timestamp_s, frame
            index += 1
    finally:
        capture.release()


def detect_source(
    source: str | Path,
    board,
    min_corners: int,
    stride: int = 1,
    refine: bool = True,
) -> dict:
    source = Path(source)
    if source.is_dir():
        frames = _iter_images_from_dir(source)
        source_kind = "images"
    else:
        frames = _iter_frames_from_video(source, stride=stride)
        source_kind = "video"

    records = []
    image_size = None
    n_seen = 0
    for frame_id, timestamp_s, image in frames:
        n_seen += 1
        if image_size is None:
            image_size = (int(image.shape[1]), int(image.shape[0]))
        detection = detect_calibration_image(image, board, refine=refine)
        if detection.n_corners < min_corners:
            continue
        records.append(
            {
                "frame": int(frame_id) if isinstance(frame_id, (int, np.integer)) else str(frame_id),
                "timestamp_s": None if timestamp_s is None else float(timestamp_s),
                "ids": detection.ids.tolist(),
                "corners": detection.corners.tolist(),
            }
        )
    if image_size is None:
        raise RuntimeError(f"No frames/images found in {source}")
    return {
        "source": str(source),
        "source_kind": source_kind,
        "board": board_metadata(board),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "n_frames_scanned": n_seen,
        "n_detections": len(records),
        "frames": records,
    }


def load_detections(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_detections(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_detection_board(payload: dict, board) -> None:
    """Reject stale detections made with a different configured target.

    Older detection files did not include board metadata, so they remain readable.
    """
    recorded = payload.get("board")
    if recorded is not None and recorded != board_metadata(board):
        raise ValueError(
            f"Detection board does not match current config: recorded={recorded}, "
            f"configured={board_metadata(board)}. Run the detect step again."
        )


def detections_to_point_lists(payload: dict, board) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    object_points = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    objs: list[np.ndarray] = []
    imgs: list[np.ndarray] = []
    meta: list[dict] = []
    for record in payload["frames"]:
        ids = np.asarray(record["ids"], dtype=np.int32).reshape(-1)
        corners = np.asarray(record["corners"], dtype=np.float32).reshape(-1, 2)
        if ids.size == 0:
            continue
        objs.append(object_points[ids].reshape(-1, 3))
        imgs.append(corners.reshape(-1, 2))
        meta.append(record)
    return objs, imgs, meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect calibration-board corners in a video or image folder")
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", choices=("A", "B"), default=None)
    parser.add_argument("--source", default=None, help="Video file or image directory")
    parser.add_argument("--output", default=None)
    parser.add_argument("--stride", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    board = calibration_board_from_config(config)
    min_corners = int(config.get("calibration", {}).get("min_corners", 8))
    stride = int(args.stride or config.get("calibration", {}).get("detect_stride", 1))
    out_dir = output_dir(config)

    jobs: list[tuple[str, Path, Path]] = []
    if args.source:
        camera = args.camera or "A"
        output = Path(args.output) if args.output else out_dir / f"detections_{camera}.json"
        jobs.append((camera, Path(args.source), output))
    else:
        cameras = ("A", "B") if args.camera is None else (args.camera,)
        for camera in cameras:
            cam_cfg = config["cameras"][camera]
            stereo_source = resolve_path(config, cam_cfg["video"])
            jobs.append((camera, stereo_source, out_dir / f"detections_{camera}.json"))
            raw_intrinsics = cam_cfg.get("intrinsics_source")
            if raw_intrinsics:
                intrinsics_source = resolve_path(config, raw_intrinsics)
                if intrinsics_source != stereo_source:
                    jobs.append(
                        (
                            camera,
                            intrinsics_source,
                            out_dir / f"detections_intrinsics_{camera}.json",
                        )
                    )

    for camera, source, output in jobs:
        payload = detect_source(source, board, min_corners=min_corners, stride=stride)
        payload["camera"] = camera
        save_detections(output, payload)
        print(
            f"{camera}: {payload['n_detections']} detections / "
            f"{payload['n_frames_scanned']} scanned from {source} -> {output}"
        )


if __name__ == "__main__":
    main()

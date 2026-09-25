"""Shared config, I/O, and ChArUco board helpers for the two-phone stereo pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only when OpenCV is absent
    cv2 = None


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"

ARUCO_DICTIONARIES = {
    "DICT_4X4_50": "DICT_4X4_50",
    "DICT_4X4_100": "DICT_4X4_100",
    "DICT_4X4_250": "DICT_4X4_250",
    "DICT_4X4_1000": "DICT_4X4_1000",
    "DICT_5X5_50": "DICT_5X5_50",
    "DICT_5X5_100": "DICT_5X5_100",
    "DICT_5X5_250": "DICT_5X5_250",
    "DICT_5X5_1000": "DICT_5X5_1000",
    "DICT_6X6_50": "DICT_6X6_50",
    "DICT_6X6_100": "DICT_6X6_100",
    "DICT_6X6_250": "DICT_6X6_250",
    "DICT_6X6_1000": "DICT_6X6_1000",
    "DICT_7X7_50": "DICT_7X7_50",
    "DICT_7X7_100": "DICT_7X7_100",
    "DICT_7X7_250": "DICT_7X7_250",
    "DICT_7X7_1000": "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL": "DICT_ARUCO_ORIGINAL",
}


@dataclass(frozen=True)
class ChessboardSpec:
    """Plain chessboard geometry, expressed using OpenCV inner-corner counts."""

    inner_corners_x: int
    inner_corners_y: int
    square_length_m: float

    @property
    def pattern_size(self) -> tuple[int, int]:
        return self.inner_corners_x, self.inner_corners_y

    def getChessboardCorners(self) -> np.ndarray:
        grid = np.zeros((self.inner_corners_x * self.inner_corners_y, 3), dtype=np.float32)
        grid[:, :2] = np.mgrid[
            0 : self.inner_corners_x, 0 : self.inner_corners_y
        ].T.reshape(-1, 2)
        grid[:, :2] *= self.square_length_m
        return grid


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} is empty or invalid")
    config["_config_path"] = str(config_path.resolve())
    config["_config_dir"] = str(config_path.resolve().parent)
    return config


def resolve_path(config: dict[str, Any], path_value: str | Path) -> Path:
    """Resolve relative paths against the repository root."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def output_dir(config: dict[str, Any]) -> Path:
    raw = config.get("paths", {}).get("outputs", DEFAULT_OUTPUT_DIR)
    directory = resolve_path(config, raw)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _to_nested(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_nested(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_nested(item) for item in value]
    return value


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_nested(data), handle, sort_keys=False, default_flow_style=None)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} is empty or invalid")
    return data


def as_matrix(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def require_cv2():
    if cv2 is None:
        raise ImportError("opencv-contrib-python is required for this step")
    return cv2


def aruco_dictionary(name: str):
    cv = require_cv2()
    key = name if name.startswith("DICT_") else f"DICT_{name}"
    if not hasattr(cv.aruco, key):
        known = ", ".join(sorted(ARUCO_DICTIONARIES))
        raise ValueError(f"Unknown ArUco dictionary {name!r}. Expected one of: {known}")
    return cv.aruco.getPredefinedDictionary(getattr(cv.aruco, key))


def charuco_board_from_config(config: dict[str, Any]):
    cv = require_cv2()
    board_cfg = config["charuco"]
    dictionary = aruco_dictionary(str(board_cfg["dictionary"]))
    return cv.aruco.CharucoBoard(
        (int(board_cfg["squares_x"]), int(board_cfg["squares_y"])),
        float(board_cfg["square_length_m"]),
        float(board_cfg["marker_length_m"]),
        dictionary,
    )


def calibration_board_from_config(config: dict[str, Any]):
    board_type = str(config.get("board", {}).get("type", "charuco")).lower()
    if board_type == "charuco":
        return charuco_board_from_config(config)
    if board_type != "chessboard":
        raise ValueError("board.type must be 'charuco' or 'chessboard'")
    board_cfg = config.get("chessboard", {})
    try:
        inner_x = int(board_cfg["inner_corners_x"])
        inner_y = int(board_cfg["inner_corners_y"])
        square_length = float(board_cfg["square_length_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "chessboard requires inner_corners_x, inner_corners_y, and square_length_m"
        ) from error
    if inner_x < 2 or inner_y < 2 or square_length <= 0:
        raise ValueError("chessboard corner counts must be >= 2 and square_length_m must be positive")
    return ChessboardSpec(inner_x, inner_y, square_length)


def board_metadata(board) -> dict[str, Any]:
    if isinstance(board, ChessboardSpec):
        return {
            "type": "chessboard",
            "inner_corners_x": board.inner_corners_x,
            "inner_corners_y": board.inner_corners_y,
            "square_length_m": board.square_length_m,
        }
    size = board.getChessboardSize()
    return {
        "type": "charuco",
        "squares_x": int(size[0]),
        "squares_y": int(size[1]),
        "square_length_m": float(board.getSquareLength()),
        "marker_length_m": float(board.getMarkerLength()),
    }


def charuco_detector(board):
    cv = require_cv2()
    return cv.aruco.CharucoDetector(board)

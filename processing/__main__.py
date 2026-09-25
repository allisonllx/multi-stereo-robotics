from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from processing.export_dataset import DatasetFrame, export_r3d, export_tum
from processing.mapping import build_coloured_map
from processing.session import decode_padded_buffer, load_frames, write_frame_sensor_associations
from processing.stereo_depth import process_sequence
from processing.trajectory import align_pair_trajectories, align_session_to_gps
from processing.trajectory import world_from_rectified_camera


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline processing for StereoCapture sessions")
    sub = parser.add_subparsers(dest="command", required=True)
    associate = sub.add_parser("associate", help="Associate GPS and heading samples with RGB frames")
    associate.add_argument("--session", required=True); associate.add_argument("--output", required=True)
    stereo = sub.add_parser("stereo-depth", help="Rectify pairs, compute depth, and compare LiDAR")
    for flag in ("session-a", "session-b", "pairs", "stereo", "output"):
        stereo.add_argument(f"--{flag}", required=True)
    stereo.add_argument("--max-pairs", type=int)
    trajectory = sub.add_parser("trajectory", help="Align Camera A ARKit trajectory to GPS ENU")
    trajectory.add_argument("--session", required=True); trajectory.add_argument("--output", required=True)
    trajectory.add_argument("--session-b"); trajectory.add_argument("--pairs"); trajectory.add_argument("--stereo"); trajectory.add_argument("--output-b")
    for name in ("export-tum", "export-r3d"):
        export = sub.add_parser(name, help=f"Create {name.removeprefix('export-').upper()} dataset")
        export.add_argument("--session", required=True); export.add_argument("--depth-dir", required=True); export.add_argument("--output", required=True)
    mapping = sub.add_parser("map", help="Fuse rectified RGB-D frames into a coloured PLY map")
    mapping.add_argument("--session", required=True); mapping.add_argument("--processed-dir", required=True); mapping.add_argument("--output", required=True)
    mapping.add_argument("--frame-stride", type=int, default=5); mapping.add_argument("--pixel-stride", type=int, default=4); mapping.add_argument("--voxel-size", type=float, default=0.05)
    all_cmd = sub.add_parser("all", help="Run depth, trajectories, map, and TUM/R3D exports")
    for flag in ("session-a", "session-b", "pairs", "stereo", "output"):
        all_cmd.add_argument(f"--{flag}", required=True)
    all_cmd.add_argument("--max-pairs", type=int)
    return parser


def _dataset_frames(session_dir: str | Path, depth_dir: str | Path) -> list[DatasetFrame]:
    root, depths = Path(session_dir), Path(depth_dir)
    processed = depths.parent
    metadata_path = processed / "stereo_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    output = []
    for frame in load_frames(root):
        depth_path = depths / f"{frame.frame_id:06d}.npy"
        if not depth_path.is_file(): continue
        depth = np.load(depth_path)
        row = frame.values
        if row.get("confidence_path") and int(row.get("confidence_width", 0)) == depth.shape[1] and int(row.get("confidence_height", 0)) == depth.shape[0]:
            confidence = decode_padded_buffer(root / row["confidence_path"], depth.shape[1], depth.shape[0], int(row["confidence_bytes_per_row"]), np.uint8)
        else:
            confidence = np.where(np.isfinite(depth) & (depth > 0), 2, 0).astype(np.uint8)
        rectified_image = processed / "rgb" / f"{frame.frame_id:06d}.jpg"
        image_path = rectified_image if rectified_image.is_file() else frame.image_path
        intrinsics = np.asarray(metadata["rectified_intrinsics"], dtype=float) if metadata else frame.intrinsics
        pose = world_from_rectified_camera(frame.camera_to_world, np.asarray(metadata["rectification_R1"])) if metadata else frame.camera_to_world
        output.append(DatasetFrame(frame.timestamp_s, image_path, depth, confidence, pose, intrinsics))
    if not output: raise ValueError(f"no numbered .npy depth maps in {depths}")
    return output


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "associate":
        print(f"Wrote {write_frame_sensor_associations(args.session, args.output)}")
    elif args.command == "stereo-depth":
        report = process_sequence(args.session_a, args.session_b, args.pairs, args.stereo, args.output, args.max_pairs)
        print(json.dumps(report, indent=2))
    elif args.command == "trajectory":
        report = align_session_to_gps(args.session, args.output)
        supplied = [args.session_b, args.pairs, args.stereo, args.output_b]
        if any(supplied) and not all(supplied):
            raise SystemExit("--session-b, --pairs, --stereo, and --output-b must be provided together")
        if all(supplied):
            report["phone_world_alignment"] = align_pair_trajectories(args.session, args.session_b, args.pairs, args.stereo, args.output_b)
        print(json.dumps(report, indent=2))
    elif args.command in {"export-tum", "export-r3d"}:
        frames = _dataset_frames(args.session, args.depth_dir)
        result = export_tum(frames, args.output) if args.command == "export-tum" else export_r3d(frames, args.output)
        print(f"Wrote {result}")
    elif args.command == "map":
        print(json.dumps(build_coloured_map(args.session, args.processed_dir, args.output, args.frame_stride, args.pixel_stride, args.voxel_size), indent=2))
    elif args.command == "all":
        output = Path(args.output); depth_output = output / "stereo"
        write_frame_sensor_associations(args.session_a, output / "associated_sensors_A.csv")
        write_frame_sensor_associations(args.session_b, output / "associated_sensors_B.csv")
        process_sequence(args.session_a, args.session_b, args.pairs, args.stereo, depth_output, args.max_pairs)
        align_session_to_gps(args.session_a, output / "trajectory_enu.csv")
        align_pair_trajectories(args.session_a, args.session_b, args.pairs, args.stereo, output / "trajectory_B_in_A_world.csv")
        frames = _dataset_frames(args.session_a, depth_output / "depth")
        export_tum(frames, output / "tum")
        export_r3d(frames, output / "dataset.r3d")
        build_coloured_map(args.session_a, depth_output, output / "map.ply")
        print(f"Wrote offline results to {output}")


if __name__ == "__main__": main()

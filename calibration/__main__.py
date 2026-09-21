"""CLI: python -m calibration <command>"""

from __future__ import annotations

import sys

from calibration.calibrate_intrinsics import main as intrinsics_main
from calibration.calibrate_stereo import main as stereo_main
from calibration.detect_charuco import main as detect_main
from calibration.extract_timestamps import main as timestamps_main
from calibration.import_session import main as import_session_main
from calibration.pair_frames import main as pair_main
from calibration.synchronize_audio import main as sync_main
from calibration.verify_calibration import main as verify_main

COMMANDS = {
    "import-session": (import_session_main, "Validate an ARKit recording and import timestamps"),
    "timestamps": (timestamps_main, "Extract per-frame presentation timestamps"),
    "sync": (sync_main, "Cross-correlate clap audio and fit t_A = a t_B + b"),
    "pair": (pair_main, "Pair Camera B frames to the nearest Camera A frame"),
    "detect": (detect_main, "Detect and subpixel-refine ChArUco corners"),
    "intrinsics": (intrinsics_main, "Calibrate each camera independently"),
    "stereo": (stereo_main, "Calibrate stereo extrinsics with fixed intrinsics"),
    "verify": (verify_main, "Rectify and write a held-out verification report"),
}

ORDER = ("timestamps", "sync", "pair", "detect", "intrinsics", "stereo", "verify")


def _print_root_help() -> None:
    print("usage: python -m calibration <command> [options]")
    print()
    print("Two-phone rigid stereo synchronization and calibration")
    print()
    print("commands:")
    print(f"  {'import-session':<12} {COMMANDS['import-session'][1]}")
    print("  all          Run the full pipeline in order")
    for name in ORDER:
        print(f"  {name:<12} {COMMANDS[name][1]}")
    print()
    print("Pass --help after a command for that step's flags, e.g.")
    print("  python -m calibration sync --help")


def _run_all(argv: list[str]) -> None:
    for name in ORDER:
        print(f"\n==> {name}")
        COMMANDS[name][0](argv)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _print_root_help()
        return
    command, rest = argv[0], argv[1:]
    if command == "all":
        _run_all(rest)
        return
    if command not in COMMANDS:
        _print_root_help()
        raise SystemExit(f"unknown command: {command}")
    COMMANDS[command][0](rest)


if __name__ == "__main__":
    main()

import pytest

from processing.__main__ import build_parser


def test_cli_parses_stereo_depth_paths():
    args = build_parser().parse_args(["stereo-depth", "--session-a", "a", "--session-b", "b", "--pairs", "pairs.csv", "--stereo", "stereo.yaml", "--output", "out"])
    assert args.command == "stereo-depth" and args.session_a == "a"


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_cli_parses_sensor_association():
    args = build_parser().parse_args(["associate", "--session", "capture", "--output", "associated.csv"])
    assert args.command == "associate" and args.session == "capture"

# Two-phone rigid stereo calibration

Synchronize two phone recordings from clap audio, then calibrate a rigidly mounted stereo pair with a printed ChArUco board.

Run every command from the **repository root**. Paths in `calibration/config.yaml` are relative to that root.

## Setup

You need:

- Python 3.10 or newer
- [FFmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on `PATH`) for audio extraction and frame timestamps
- OpenCV contrib (ArUco / ChArUco live there, not in plain `opencv-python`)

```bash
# macOS
brew install ffmpeg

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check the install:

```bash
ffmpeg -version
ffprobe -version
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"
python -m calibration --help
python -m pytest tests -q
```

`opencv-contrib-python` and `opencv-python` cannot sit in the same environment. If `cv2.aruco` is missing, uninstall both and reinstall only `opencv-contrib-python`.

## Capture (before you feed anything in)

Lock the rig and the recording settings first:

- Both phones firmly in the handle; do not move them after extrinsic calibration
- Same resolution and FPS you will use in the experiment
- One fixed lens; digital stabilization off
- Lock focus, exposure, and white balance if the app allows it

**Claps (required for sync).** At the beginning *and* the end of each paired recording:

1. Put your hands in both cameras’ shared field of view
2. Do three clearly spaced claps
3. Confirm both phones recorded the audio

**ChArUco board.** Prefer a print on rigid foam board, not a laptop screen. Record in `calibration/config.yaml`:

| Field | Meaning |
|---|---|
| `squares_x`, `squares_y` | Number of chessboard **squares** along each side (not inner corners) |
| `square_length_m` | Side length of one square, metres |
| `marker_length_m` | Side length of the ArUco marker inside a square, metres |
| `dictionary` | Must match the print, e.g. `DICT_5X5_250` |

For stereo poses, hold the board still for about one second at each position. That way a one-frame sync error still sees the same geometry.

Intrinsics belong to each phone. Extrinsics belong to the assembled rig. If you can, capture ~20–30 intrinsic views per phone (board through the image: centre, corners, near/far, tilted), then a separate stereo take with both phones mounted (~15–25 still poses).

## How to feed in input

### File layout

```text
robotics/
├── data/
│   ├── camera_A.mp4              # stereo take, phone A (video + audio)
│   ├── camera_B.mp4              # stereo take, phone B (video + audio)
│   ├── intrinsics_A/             # optional: stills or a dedicated video
│   └── intrinsics_B/
└── calibration/
    └── config.yaml               # edit this
```

Copy or rename your files to match, or change the paths in config. Either is fine.

### Video format

| Requirement | Detail |
|---|---|
| Container | Anything FFmpeg can decode: `.mp4`, `.mov`, `.m4v`, `.mkv`, … |
| Video codec | H.264 is the usual phone default and is fine |
| Audio | **Required** on both stereo files. Mono or stereo. Claps must be audible |
| Timebase | Variable frame rate is OK. The pipeline reads presentation timestamps; it does not assume `t = i / FPS` |
| Resolution / FPS | Same on both phones, and the same as the experiment |

Audio-less clips cannot be synchronized. If a file has no soundtrack, re-export it from the phone with audio enabled.

Name the phones consistently: **A** is the reference timeline. Every Camera B timestamp is mapped onto Camera A with `t_A = a t_B + b`.

### Optional intrinsic stills

`intrinsics_source` may be a **video file** or a **folder of images**. Image folders are read in sorted filename order. Supported stills:

`.png` `.jpg` `.jpeg` `.bmp` `.tif` `.tiff` `.webp`

If you skip `intrinsics_source`, or point it at the same file as `video`, intrinsics are estimated from the stereo take.

### Edit `calibration/config.yaml`

```yaml
cameras:
  A:
    video: data/camera_A.mp4
    intrinsics_source: data/camera_A.mp4    # or data/intrinsics_A/ or a dedicated .mp4
  B:
    video: data/camera_B.mp4
    intrinsics_source: data/camera_B.mp4

charuco:
  squares_x: 7
  squares_y: 5
  square_length_m: 0.040
  marker_length_m: 0.030
  dictionary: DICT_5X5_250

verification:
  measured_baseline_m: 0.12   # tape measure between optical centres; optional
```

Paths are relative to the **repo root**, not to `calibration/`. Absolute paths also work.

Useful knobs:

- `sync.clap_window_s` — first/last seconds used to find claps (default 8)
- `sync.drift_threshold_ms` — if start vs end lag differs by more than this, fit clock-rate `a`
- `calibration.pose_interval_s` — keep at most one stereo pose per this many seconds (hold-still sampling)
- `calibration.min_corners` — drop frames with fewer shared ChArUco corners

## Run

From the repo root, after videos exist and config is edited:

```bash
python -m calibration all
```

Or one step at a time:

```bash
python -m calibration timestamps   # PTS for every frame
python -m calibration sync         # clap cross-correlation → a, b
python -m calibration pair         # nearest-frame pairs + clap frame numbers
python -m calibration detect       # ChArUco corners
python -m calibration intrinsics   # K and distortion per camera
python -m calibration stereo       # R, T, E, F with intrinsics held fixed
python -m calibration verify       # held-out diagnostics + rectified preview
```

Pass `--config path/to/config.yaml` to any step. `python -m calibration sync --help` lists flags for that step.

You can override files on the command line without editing YAML, for example:

```bash
python -m calibration sync --video-a /path/to/A.mov --video-b /path/to/B.mov
python -m calibration detect --camera A --source data/intrinsics_A/
```

## Outputs

Written to `calibration/outputs/` (overridable via `paths.outputs`):

| File | What it is |
|---|---|
| `timestamps_A.csv`, `timestamps_B.csv` | `frame, timestamp_s` from the container PTS |
| `time_mapping.json` | `a`, `b`, start/end lag, detected clap times |
| `synchronization.csv` | `frame_A, timestamp_A, frame_B, timestamp_B, residual_ms` |
| `detections_A.json`, `detections_B.json` | Per-frame ChArUco IDs and corners (stereo video) |
| `detections_intrinsics_A.json` | Only if `intrinsics_source` is a different path |
| `intrinsics_A.yaml`, `intrinsics_B.yaml` | Image size, `K`, distortion, RMS reprojection error |
| `stereo_extrinsics.yaml` | `R`, `T`, `E`, `F`, baseline `‖T‖` |
| `verification_report.json` | Epipolar distance, vertical disparity, triangulated square length |
| `rectified_preview.png` | Side-by-side rectified frames with horizontal lines |

After `pair`, the console prints clap frame indices. Open those frames on both phones and confirm the hands meet. Synchronization is **nearest-frame**, not simultaneous exposure. Residuals should stay below one frame interval (about 33 ms at 30 FPS).

Diagnostic targets, not guarantees:

- Intrinsic RMS ≲ 0.5 px
- Held-out epipolar or rectified vertical error ≲ 1 px
- Triangulated ChArUco square length close to `square_length_m`
- `‖T‖` close to `measured_baseline_m` if you set it

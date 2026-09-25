# Two-phone rigid stereo calibration

Synchronize two phone recordings from clap audio, then calibrate a rigidly mounted stereo pair with either a printed ChArUco board or a plain chessboard.

Run every command from the **repository root**. Paths in `calibration/config.yaml` are relative to that root.

## Setup

You need:

- Python 3.10 or newer
- [FFmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on `PATH`) for audio extraction and frame timestamps
- OpenCV contrib (required for ChArUco and the robust chessboard detector)

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

## ARKit recorder (iPhone 17 Pro + iPhone 15)

The native recorder is in [`ios/StereoCapture.xcodeproj`](ios/StereoCapture.xcodeproj).
Open it in Xcode, select your Personal Team in Signing & Capabilities, and run
the same build on both phones. Use one shared session ID and assign rig roles A
and B. The app records all streams during one Start/Stop run:

- RGB, ARKit pose and per-frame intrinsics at a selectable 5, 10, or 15 saved frames/s
- LiDAR scene depth and confidence on supported phones
- GPS and compass samples at their native, irregular update rates
- one continuous 48 kHz audio track for clap synchronization

ARKit continues tracking at a supported native 30 fps even when images are
saved at 10 fps. Both phones request the same manual profile: 1/120-second
exposure, ISO 100, 5000 K white balance, and lens position 0.75. The achieved
values are stored on every frame because different camera modules can clamp or
interpret the same request differently. After stopping, export each session folder to the Mac. Import
the pair together so their session IDs and rig roles are checked:

```bash
python -m calibration import-session /path/to/session_A /path/to/session_B
```

This writes `timestamps_A.csv`, `timestamps_B.csv`, and a validation report for
each phone under `calibration/outputs/`. The frame timestamps share the audio
timeline, so the existing clap synchronizer can operate directly on the two
exported audio files:

```bash
python -m calibration sync \
  --video-a /path/to/session_A/audio.m4a \
  --video-b /path/to/session_B/audio.m4a
python -m calibration pair
python -m calibration detect --camera A --source /path/to/session_A/rgb
python -m calibration detect --camera B --source /path/to/session_B/rgb
python -m calibration intrinsics
python -m calibration stereo
python -m calibration verify
```

For intrinsic calibration from these image folders, set each camera's
`intrinsics_source` in `calibration/config.yaml` to its exported `rgb/` folder.
Do not run `timestamps` after importing an ARKit session; it would try to treat
the image folders as video and replace the imported timestamps.

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

**Calibration board.** Prefer a print on rigid foam board, not a laptop screen. Set `board.type` to `charuco` or `chessboard` in `calibration/config.yaml`.

For ChArUco:

| Field | Meaning |
|---|---|
| `squares_x`, `squares_y` | Number of chessboard **squares** along each side (not inner corners) |
| `square_length_m` | Side length of one square, metres |
| `marker_length_m` | Side length of the ArUco marker inside a square, metres |
| `dictionary` | Must match the print, e.g. `DICT_5X5_250` |

For a plain chessboard, set `inner_corners_x`, `inner_corners_y`, and `square_length_m`. The first two are the number of **internal corner intersections**, not the number of squares. A non-square pattern such as 9×6 is recommended; keep the board orientation similar in both cameras because a markerless board cannot encode a unique physical origin.

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

board:
  type: charuco             # or chessboard

charuco:
  squares_x: 7
  squares_y: 5
  square_length_m: 0.040
  marker_length_m: 0.030
  dictionary: DICT_5X5_250

# Used only when board.type is chessboard:
chessboard:
  inner_corners_x: 10
  inner_corners_y: 7
  square_length_m: 0.0275  # 2.75 cm

verification:
  measured_baseline_m: 0.12   # tape measure between optical centres; optional
```

Paths are relative to the **repo root**, not to `calibration/`. Absolute paths also work.

Useful knobs:

- `sync.clap_window_s` — first/last seconds used to find claps (default 8)
- `sync.drift_threshold_ms` — if start vs end lag differs by more than this, fit clock-rate `a`
- `calibration.pose_interval_s` — keep at most one stereo pose per this many seconds (hold-still sampling)
- `calibration.min_corners` — drop frames with fewer shared board corners

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
python -m calibration detect       # ChArUco or plain chessboard corners
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
| `detections_A.json`, `detections_B.json` | Per-frame board IDs and corners (stereo video) |
| `detections_intrinsics_A.json` | Only if `intrinsics_source` is a different path |
| `intrinsics_A.yaml`, `intrinsics_B.yaml` | Image size, `K`, distortion, RMS reprojection error |
| `stereo_extrinsics.yaml` | `R`, `T`, `E`, `F`, baseline `‖T‖` |
| `verification_report.json` | Epipolar distance, vertical disparity, triangulated square length |
| `rectified_preview.png` | Side-by-side rectified frames with horizontal lines |

After `pair`, the console prints clap frame indices. Open those frames on both phones and confirm the hands meet. Synchronization is **nearest-frame**, not simultaneous exposure. Residuals should stay below one frame interval (about 33 ms at 30 FPS).

Diagnostic targets, not guarantees:

- Intrinsic RMS ≲ 0.5 px
- Held-out epipolar or rectified vertical error ≲ 1 px
- Triangulated board square length close to `square_length_m`
- `‖T‖` close to `measured_baseline_m` if you set it

## Offline depth, trajectory, and dataset processing

After importing and synchronizing the two sessions and completing calibration,
run the combined offline stage:

```bash
python -m processing all \
  --session-a /path/to/session_A \
  --session-b /path/to/session_B \
  --pairs calibration/outputs/synchronization.csv \
  --stereo calibration/outputs/stereo_extrinsics.yaml \
  --output processing/outputs/walk_001
```

This writes StereoSGBM depth (`stereo/depth/*.npy`), per-frame and aggregate
LiDAR comparison metrics, a GPS-aligned Camera A trajectory in local ENU metres,
and a TUM/OpenLORIS-style dataset plus a voxel-downsampled coloured `map.ply`.
Run stages independently with:

```bash
python -m processing stereo-depth --help
python -m processing trajectory --help
python -m processing export-tum --help
python -m processing export-r3d --help
python -m processing map --help
```

R3D export requires `lzfse`, included in `requirements.txt`. Its archive follows
the CROSS loader's exact `metadata` plus `rgbd/{i}.jpg/.depth/.conf` contract.
CROSS currently reconstructs R3D timestamps as `index / fps`, so use the TUM
export when retaining irregular real timestamps is important.

Coordinate conventions:

- ARKit poses are camera-to-world transforms with metres.
- Stereo calibration stores `X_B = R X_A + T`.
- Camera A is the rig reference.
- GPS is converted to local East-North-Up and fitted to ARKit's horizontal
  `(x, -z)` path with rotation and translation only—no scale correction.
- Compass remains an observed prior; it is not averaged directly into ARKit.

`map.ply` is direct RGB-D fusion using the recorded ARKit poses and voxel
downsampling. It does not perform place recognition or pose-graph loop closure;
those require a separate SLAM backend and should only be added after evaluating
the ARKit trajectory drift on the first indoor/outdoor loop captures.

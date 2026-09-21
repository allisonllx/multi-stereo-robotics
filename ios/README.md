# Stereo Capture iPhone app

Open `StereoCapture.xcodeproj` in Xcode, choose your Personal Team under
Signing & Capabilities, connect an iPhone, and press Run. Install the same build
on both phones. Enter the same session ID and assign one phone A and the other B.

The app runs ARKit at a supported native 30 fps format and saves 5, 10, or 15
RGB frames per second. GPS, heading, audio, and ARKit callbacks remain independent
and are timestamped within the same Start/Stop recording.

Both phones request the same locked capture profile: 1/120-second exposure,
ISO 100, 5000 K white balance, and lens position 0.75. Every frame also records
the actual exposure, ISO, focus position, and white-balance gains achieved by
that device. Equal numeric focus positions do not imply exactly equal focus
distance or field of view across different iPhone camera modules; calibration
still remains phone- and format-specific.

`frames.csv.timestamp_s` uses the audio-recorder timeline so clap times from
`audio.m4a` can be applied directly. `arkit_timestamp_s` retains the original
monotonic ARKit clock, and `unix_time_s` provides approximate wall-clock time.

The iPhone 17 Pro also writes raw `Float32` scene-depth buffers and `UInt8`
confidence buffers. Each frame row records their width, height, and byte stride
so the padded raw buffers can be decoded reliably offline.

After stopping, use **Export Session** and transfer the folder to the Mac. Then:

```bash
python -m calibration import-session /path/to/session_A /path/to/session_B
```

The two exports must retain their complete folder structure.

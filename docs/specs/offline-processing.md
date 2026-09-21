# Offline Processing Specification

Build a deterministic offline pipeline for paired StereoCapture exports. It must:

1. Decode row-padded Float32 LiDAR depth and UInt8 confidence buffers.
2. Associate GPS, heading, and other timestamped samples with RGB frames without pretending the sensors share a rate.
3. Rectify synchronized RGB pairs with the existing stereo calibration, calculate StereoSGBM disparity/depth, and emit confidence-aware comparison metrics against LiDAR.
4. Parse per-frame ARKit camera-to-world poses, transform Camera B into Camera A using the fixed stereo extrinsic, and align the metric local trajectory to GPS in a local East-North-Up frame.
5. Export TUM/OpenLORIS-style datasets and CROSS-compatible R3D archives while retaining the immutable source sessions.

All generated artifacts belong under a caller-selected output directory, use metres and seconds, and must be excluded from Git when written under `processing/outputs/`.


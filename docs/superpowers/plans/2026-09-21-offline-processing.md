# Offline Stereo and Mapping Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn two validated StereoCapture exports into synchronized sensor samples, rectified stereo depth, LiDAR comparison metrics, GPS-anchored trajectories, and CROSS-compatible TUM/R3D datasets.

**Architecture:** Keep immutable phone sessions as inputs and put each transformation in a focused Python module. A CLI command per stage writes inspectable NumPy, PNG, CSV, JSON, and dataset artifacts; a final `process` command composes the stages after synchronization and calibration outputs exist.

**Tech Stack:** Python 3.10+, NumPy, SciPy, OpenCV contrib, PyYAML, standard-library CSV/JSON/ZIP.

**Spec:** `docs/specs/offline-processing.md`

## Global Constraints

- Timestamp units are seconds; geometric units are metres.
- Camera poses are 4×4 camera-to-world transforms stored row-major in `frames.csv`.
- Camera A is the rig reference; stereo calibration maps Camera A coordinates to Camera B as `X_B = R X_A + T`.
- Raw session folders are never modified.
- GPS and heading are associated by time and retain age/accuracy metadata.
- R3D export documents that CROSS currently synthesizes timestamps as `index/fps`.

---

### Task 1: Session streams and LiDAR decoding

**Files:**
- Create: `processing/session.py`
- Create: `tests/test_processing_session.py`

**Interfaces:**
- Produces: `load_frames(path) -> list[FrameRecord]`, `decode_padded_buffer(path, width, height, bytes_per_row, dtype) -> np.ndarray`, `associate_nearest(query_times, sample_times, max_delta_s=None) -> Association`.

- [ ] Write tests for padded Float32/UInt8 decoding, pose parsing, nearest association, and stale-sample rejection.
- [ ] Run `python -m pytest tests/test_processing_session.py -q` and confirm missing-module failure.
- [ ] Implement immutable records, strict CSV validation, row-stride removal, and vectorized nearest association.
- [ ] Run the focused tests and confirm they pass.
- [ ] Commit the independently working stream layer.

### Task 2: Stereo rectification, disparity, and LiDAR evaluation

**Files:**
- Create: `processing/stereo_depth.py`
- Create: `tests/test_stereo_depth.py`

**Interfaces:**
- Consumes: existing `rectification.yaml`, `stereo_extrinsics.yaml`, synchronization pairs, and Task 1 depth decoder.
- Produces: `disparity_to_depth(disparity, focal_px, baseline_m)`, `evaluate_depth(estimate, reference, valid_mask)`, and `process_stereo_pair(...)`.

- [ ] Write synthetic geometry tests for depth conversion, invalid disparity masking, and metric summaries.
- [ ] Run the focused tests and confirm missing-module failure.
- [ ] Implement rectification-map loading, StereoSGBM disparity, metric depth, confidence filtering, and per-frame aggregate JSON output.
- [ ] Run focused tests and a synthetic image smoke test.
- [ ] Commit the depth stage.

### Task 3: ARKit trajectories and GPS anchoring

**Files:**
- Create: `processing/trajectory.py`
- Create: `tests/test_trajectory.py`

**Interfaces:**
- Produces: `camera_b_to_a_world_pose`, `geodetic_to_enu`, `fit_planar_rigid_alignment`, and `write_aligned_trajectory`.

- [ ] Write tests for the stereo pose relation, known ENU displacements, and a known 2D rigid transform recovered from paired points.
- [ ] Run focused tests and confirm missing-module failure.
- [ ] Implement pose composition, WGS84 local ENU conversion, accuracy filtering, timestamp association, and planar SVD alignment without scale.
- [ ] Run focused tests and confirm they pass.
- [ ] Commit the trajectory stage.

### Task 4: TUM and R3D export

**Files:**
- Create: `processing/export_dataset.py`
- Create: `tests/test_export_dataset.py`

**Interfaces:**
- Consumes: Camera A RGB, aligned metric depth, confidence, timestamps, intrinsics, and world poses.
- Produces: `export_tum(...)` and `export_r3d(...)`.

- [ ] Write tests that inspect exact filenames, TUM text columns, millimetre PNG depth, R3D metadata keys, pose quaternion order, and archive members.
- [ ] Run focused tests and confirm missing-module failure.
- [ ] Implement OpenLORIS-style `color.txt`, `aligned_depth.txt`, `groundtruth.txt`, `odom.txt`, OpenCV YAML calibration, and CROSS R3D ZIP layout with LZFSE availability checked explicitly.
- [ ] Run focused tests and validate the generated archive structure.
- [ ] Commit the export stage.

### Task 5: CLI composition and documentation

**Files:**
- Create: `processing/__init__.py`
- Create: `processing/__main__.py`
- Modify: `README.md`
- Modify: `.gitignore`
- Test: `tests/test_processing_cli.py`

**Interfaces:**
- Produces commands `python -m processing associate|stereo-depth|trajectory|export-tum|export-r3d|all`.

- [ ] Write CLI parser and dry-run validation tests.
- [ ] Run focused tests and confirm failure before implementation.
- [ ] Wire stage functions with explicit paths and actionable missing-input errors; document commands and coordinate conventions.
- [ ] Run all Python tests and CLI help smoke tests.
- [ ] Build the iOS app for simulator and device to ensure no regression.
- [ ] Commit the complete offline pipeline.


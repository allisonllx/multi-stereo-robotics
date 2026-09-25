# Capture and Offline Readiness Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the known capture-integrity, coordinate-frame, synchronization, GPS, LiDAR-role, and long-map memory blockers before physical two-phone trials.

**Architecture:** Keep camera-frame transforms explicit through small tested helpers, carry rectified and ENU poses in processing metadata, and make optional sensors degrade without aborting geometry-only outputs. On iOS, use a bounded serial writer queue that owns file handles and reports drops/errors back to the UI.

**Tech Stack:** Swift/ARKit/AVFoundation, Python 3.10+, NumPy, SciPy, OpenCV, pytest, Xcode 16.

**Spec:** `docs/specs/offline-processing.md` plus the readiness findings recorded in the 2026-09-25 review.

## Global Constraints

- Camera A remains the stereo reference, but either role may contain LiDAR.
- OpenCV camera axes are `(right, down, forward)`; ARKit camera axes are `(right, up, backward)`.
- Generated RGB/depth and their camera-to-world pose must describe the same rectified optical frame.
- GPS is optional; absence must not block stereo, mapping, or dataset export.
- Phone capture may drop work under pressure, but it must count and report every drop or write failure.

---

### Task 1: Coordinate and rectification correctness

**Files:** `processing/trajectory.py`, `processing/stereo_depth.py`, `processing/mapping.py`, `processing/__main__.py`, `tests/test_trajectory.py`, `tests/test_stereo_depth.py`

**Interfaces:** Produce `opencv_extrinsic_to_arkit()`, `world_from_rectified_camera()`, and rectified per-frame poses in `stereo_metadata.json`.

- [ ] Write failing synthetic transform tests covering axis conjugation and rectification rotation.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement the helpers and use them for phone-world alignment, map fusion, and exports.
- [ ] Run focused tests and commit.

### Task 2: Safe stereo pairing and either-role LiDAR

**Files:** `calibration/pair_frames.py`, `processing/stereo_depth.py`, `processing/__main__.py`, `tests/test_pair_frames.py`, `tests/test_stereo_depth.py`

**Interfaces:** Produce filtered one-to-one `FramePair` lists and `process_sequence(..., lidar_camera="auto")`.

- [ ] Write failing tests for duplicate rejection, maximum residual filtering, and Camera B LiDAR selection.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement pairing filters and role-aware LiDAR rectification into Camera A's rectified plane.
- [ ] Run focused tests and commit.

### Task 3: Optional GPS and pose propagation

**Files:** `processing/trajectory.py`, `processing/__main__.py`, `processing/export_dataset.py`, `tests/test_trajectory.py`, `tests/test_processing_cli.py`

**Interfaces:** Produce an optional world transform from ARKit world to ENU and apply it to dataset/map poses when available.

- [ ] Write failing tests for missing GPS, unique-fix fitting, and ENU pose transformation.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement optional GPS behavior, unique-fix weighting, robust spread checks, and pose propagation.
- [ ] Run focused tests and commit.

### Task 4: Bounded map fusion

**Files:** `processing/mapping.py`, `tests/test_mapping.py`

**Interfaces:** Produce `VoxelAccumulator.add(points, colours)` with bounded voxel state rather than retaining all raw points.

- [ ] Write a failing multi-batch equivalence test.
- [ ] Run it and confirm failure.
- [ ] Implement incremental voxel sums/counts and update map building.
- [ ] Run focused tests and commit.

### Task 5: Bounded iOS capture writer

**Files:** `ios/StereoCapture/SessionRecorder.swift`, `ios/StereoCapture/ContentView.swift`, `ios/README.md`

**Interfaces:** A bounded serial writer queue accepts immutable frame jobs, owns file handles, exposes queued/dropped/failed counters, drains before final manifest creation, and performs centralized teardown.

- [ ] Add observable drop/failure status to the UI and implement a bounded pending-job counter.
- [ ] Move JPEG/raw-buffer/CSV writes off `MainActor`, propagate errors, verify permissions/audio startup, and centralize cleanup.
- [ ] Build simulator and device targets and perform a source audit for remaining synchronous frame writes.
- [ ] Commit.

### Task 6: Full verification and documentation

**Files:** `README.md`, existing tests.

- [ ] Document role-independent LiDAR, optional GPS, coordinate conventions, and capture health counters.
- [ ] Run all Python tests, CLI smoke tests, simulator build, and device build.
- [ ] Review the diff for the five original readiness findings and commit final documentation.

import ARKit
import AVFoundation
import CoreImage
import CoreLocation
import SwiftUI
import UIKit

private struct WriterSummary {
    let savedFrames: Int
    let failures: Int
}

private final class FrameWriteJob: @unchecked Sendable {
    let id: Int
    let csvLine: String
    let image: CVPixelBuffer
    let depth: CVPixelBuffer?
    let confidence: CVPixelBuffer?

    init(id: Int, csvLine: String, image: CVPixelBuffer, depth: CVPixelBuffer?, confidence: CVPixelBuffer?) {
        self.id = id
        self.csvLine = csvLine
        self.image = image
        self.depth = depth
        self.confidence = confidence
    }
}

private final class CaptureWriter: @unchecked Sendable {
    private let root: URL
    private let queue = DispatchQueue(label: "stereo.capture.writer", qos: .userInitiated)
    private let lock = NSLock()
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let framesFile: FileHandle
    private let locationsFile: FileHandle
    private let headingsFile: FileHandle
    private let maxPendingFrames: Int
    private var pendingFrames = 0
    private var savedFrames = 0
    private var failures = 0
    private var isFinishing = false

    init(root: URL, framesHeader: String, locationHeader: String, headingHeader: String,
         maxPendingFrames: Int = 3) throws {
        self.root = root
        self.maxPendingFrames = maxPendingFrames
        framesFile = try Self.makeCSV(root.appendingPathComponent("frames.csv"), header: framesHeader)
        locationsFile = try Self.makeCSV(root.appendingPathComponent("location.csv"), header: locationHeader)
        headingsFile = try Self.makeCSV(root.appendingPathComponent("heading.csv"), header: headingHeader)
    }

    func enqueue(_ job: FrameWriteJob, completion: @escaping @Sendable (WriterSummary) -> Void) -> Bool {
        lock.lock()
        guard !isFinishing, pendingFrames < maxPendingFrames else {
            lock.unlock()
            return false
        }
        pendingFrames += 1
        queue.async { [self] in
            do {
                try write(job)
                savedFrames += 1
            } catch {
                failures += 1
            }
            lock.lock()
            pendingFrames -= 1
            let summary = WriterSummary(savedFrames: savedFrames, failures: failures)
            lock.unlock()
            completion(summary)
        }
        lock.unlock()
        return true
    }

    func enqueueLocation(_ line: String) { enqueueSensor(line, handle: locationsFile) }
    func enqueueHeading(_ line: String) { enqueueSensor(line, handle: headingsFile) }

    func finish() throws -> WriterSummary {
        lock.lock()
        isFinishing = true
        lock.unlock()
        return try queue.sync {
            try framesFile.synchronize()
            try locationsFile.synchronize()
            try headingsFile.synchronize()
            try framesFile.close()
            try locationsFile.close()
            try headingsFile.close()
            return WriterSummary(savedFrames: savedFrames, failures: failures)
        }
    }

    private func enqueueSensor(_ line: String, handle: FileHandle) {
        lock.lock()
        guard !isFinishing else {
            lock.unlock()
            return
        }
        queue.async { [self] in
            do { try Self.append(line, to: handle) }
            catch { failures += 1 }
        }
        lock.unlock()
    }

    private func write(_ job: FrameWriteJob) throws {
        let stem = String(format: "%06d", job.id)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let jpeg = ciContext.jpegRepresentation(
                of: CIImage(cvPixelBuffer: job.image), colorSpace: colorSpace, options: [:]) else {
            throw NSError(domain: "StereoCapture.Writer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Could not encode frame \(job.id) as JPEG."])
        }
        try jpeg.write(to: root.appendingPathComponent("rgb/\(stem).jpg"), options: .atomic)
        if let depth = job.depth {
            try Self.writePixelBuffer(depth, to: root.appendingPathComponent("depth/\(stem).depth"))
        }
        if let confidence = job.confidence {
            try Self.writePixelBuffer(confidence, to: root.appendingPathComponent("confidence/\(stem).conf"))
        }
        // Only publish a frame row after all of its referenced files exist.
        try Self.append(job.csvLine, to: framesFile)
    }

    private static func makeCSV(_ url: URL, header: String) throws -> FileHandle {
        guard FileManager.default.createFile(atPath: url.path, contents: Data((header + "\n").utf8)) else {
            throw CocoaError(.fileWriteUnknown)
        }
        return try FileHandle(forWritingTo: url)
    }

    private static func append(_ line: String, to handle: FileHandle) throws {
        try handle.write(contentsOf: Data((line + "\n").utf8))
    }

    private static func writePixelBuffer(_ buffer: CVPixelBuffer, to url: URL) throws {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw NSError(domain: "StereoCapture.Writer", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Pixel buffer has no readable storage."])
        }
        try Data(bytes: base, count: CVPixelBufferGetDataSize(buffer)).write(to: url, options: .atomic)
    }
}

@MainActor
final class SessionRecorder: NSObject, ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var savedFrames = 0
    @Published private(set) var droppedFrames = 0
    @Published private(set) var writeFailures = 0
    @Published private(set) var trackingState = "not started"
    @Published private(set) var locationStatus = "waiting"
    @Published private(set) var hasLiDAR = false
    @Published private(set) var statusMessage = "Enter the same session ID on both phones."
    @Published private(set) var lastError: Error?
    @Published private(set) var completedSessionURL: URL?

    private let arSession = ARSession()
    var previewSession: ARSession { arSession }
    private let locationManager = CLLocationManager()
    private var sessionRoot: URL?
    private var writer: CaptureWriter?
    private var audioRecorder: AVAudioRecorder?
    private var frameID = 0
    private var lastSavedTimestamp = -Double.infinity
    private var sessionID = ""
    private var role = "A"
    private var targetSavedFPS = 10
    private var saveInterval = 0.1
    private var captureDevice: AVCaptureDevice?
    private let requestedExposureSeconds = 1.0 / 120.0
    private let requestedISO: Float = 100
    private let requestedLensPosition: Float = 0.75
    private let requestedWhiteBalanceTemperature: Float = 5000

    override init() {
        super.init()
        arSession.delegate = self
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.distanceFilter = kCLDistanceFilterNone
        locationManager.headingFilter = 1
    }

    func prepare() {
        locationManager.requestWhenInUseAuthorization()
        AVCaptureDevice.requestAccess(for: .video) { _ in }
        AVAudioApplication.requestRecordPermission { _ in }
        hasLiDAR = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        let configuration = ARWorldTrackingConfiguration()
        configuration.videoHDRAllowed = false
        if let format = Self.preferredVideoFormat() { configuration.videoFormat = format }
        if hasLiDAR { configuration.frameSemantics.insert(.sceneDepth) }
        arSession.run(configuration)
    }

    func start(sessionID: String, role: String, savedFPS: Int) {
        guard !isRecording else { return }
        do {
            try validatePermissions()
            self.sessionID = sessionID.trimmingCharacters(in: .whitespacesAndNewlines)
            self.role = role
            self.targetSavedFPS = [5, 10, 15].contains(savedFPS) ? savedFPS : 10
            self.saveInterval = 1.0 / Double(self.targetSavedFPS)
            let root = try makeSessionDirectory(sessionID: self.sessionID, role: role)
            sessionRoot = root
            writer = try CaptureWriter(root: root, framesHeader: Self.framesHeader,
                                       locationHeader: Self.locationHeader, headingHeader: Self.headingHeader)
            try startAudio(at: root.appendingPathComponent("audio.m4a"))

            frameID = 0
            savedFrames = 0
            droppedFrames = 0
            writeFailures = 0
            lastSavedTimestamp = -.infinity
            completedSessionURL = nil
            lastError = nil
            isRecording = true
            if Self.locationAuthorized(locationManager.authorizationStatus) {
                locationStatus = "waiting"
                locationManager.startUpdatingLocation()
                locationManager.startUpdatingHeading()
            } else {
                locationStatus = "disabled"
            }

            let configuration = ARWorldTrackingConfiguration()
            configuration.worldAlignment = .gravity
            configuration.providesAudioData = false
            configuration.videoHDRAllowed = false
            if let format = Self.preferredVideoFormat() { configuration.videoFormat = format }
            if hasLiDAR { configuration.frameSemantics.insert(.sceneDepth) }
            arSession.run(configuration, options: [.resetTracking, .removeExistingAnchors])
            try applyManualCameraProfile()
            statusMessage = "Recording Camera \(role) at \(targetSavedFPS) saved fps. Clap where both phones can see and hear it."
        } catch {
            fail(error)
        }
    }

    func stop() {
        guard isRecording else { return }
        isRecording = false
        do {
            let summary = try teardownCapture()
            savedFrames = summary.savedFrames
            writeFailures = summary.failures
            guard let root = sessionRoot else { return }
            try writeManifest(at: root)
            completedSessionURL = root
            statusMessage = "Saved \(savedFrames) frames; dropped \(droppedFrames); write failures \(writeFailures). Export the session before deleting the app."
        } catch { fail(error) }
    }

    private func validatePermissions() throws {
        var missing: [String] = []
        if AVCaptureDevice.authorizationStatus(for: .video) != .authorized { missing.append("Camera") }
        if AVAudioApplication.shared.recordPermission != .granted { missing.append("Microphone") }
        guard missing.isEmpty else {
            throw NSError(domain: "StereoCapture", code: 10, userInfo: [
                NSLocalizedDescriptionKey: "Allow \(missing.joined(separator: ", ")) access in Settings, then try again."
            ])
        }
    }

    private static func locationAuthorized(_ status: CLAuthorizationStatus) -> Bool {
        status == .authorizedAlways || status == .authorizedWhenInUse
    }

    private func teardownCapture() throws -> WriterSummary {
        arSession.pause()
        locationManager.stopUpdatingLocation()
        locationManager.stopUpdatingHeading()
        audioRecorder?.stop()
        audioRecorder = nil
        try? AVAudioSession.sharedInstance().setActive(false)
        let activeWriter = writer
        writer = nil
        return try activeWriter?.finish() ?? WriterSummary(savedFrames: savedFrames, failures: writeFailures)
    }

    private static func preferredVideoFormat() -> ARConfiguration.VideoFormat? {
        let formats = ARWorldTrackingConfiguration.supportedVideoFormats
        return formats.first { $0.framesPerSecond == 30 && $0.imageResolution.width >= 1920 }
            ?? formats.first { $0.framesPerSecond == 30 }
            ?? formats.first
    }

    private func applyManualCameraProfile() throws {
        guard let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera else {
            throw NSError(domain: "StereoCapture", code: 1, userInfo: [NSLocalizedDescriptionKey: "ARKit did not expose a configurable primary camera."])
        }
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }

        if device.isExposureModeSupported(.custom) {
            let format = device.activeFormat
            let requested = requestedExposureSeconds
            let seconds = min(max(requested, format.minExposureDuration.seconds), format.maxExposureDuration.seconds)
            let iso = min(max(requestedISO, format.minISO), format.maxISO)
            device.setExposureModeCustom(duration: CMTime(seconds: seconds, preferredTimescale: 1_000_000_000), iso: iso)
        }
        if device.isFocusModeSupported(.locked) {
            device.setFocusModeLocked(lensPosition: requestedLensPosition)
        }
        if device.isWhiteBalanceModeSupported(.locked) {
            let desired = device.deviceWhiteBalanceGains(for: .init(temperature: requestedWhiteBalanceTemperature, tint: 0))
            let maximum = device.maxWhiteBalanceGain
            let gains = AVCaptureDevice.WhiteBalanceGains(
                redGain: min(max(desired.redGain, 1), maximum),
                greenGain: min(max(desired.greenGain, 1), maximum),
                blueGain: min(max(desired.blueGain, 1), maximum)
            )
            device.setWhiteBalanceModeLocked(with: gains)
        }
        captureDevice = device
    }

    private func makeSessionDirectory(sessionID: String, role: String) throws -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let safeID = sessionID.replacingOccurrences(of: "[^A-Za-z0-9_-]", with: "-", options: .regularExpression)
        let root = documents.appendingPathComponent("\(safeID)_\(role)_\(Int(Date().timeIntervalSince1970))", isDirectory: true)
        for name in ["rgb", "depth", "confidence"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        return root
    }

    private func startAudio(at url: URL) throws {
        let audio = AVAudioSession.sharedInstance()
        try audio.setCategory(.record, mode: .measurement)
        try audio.setActive(true)
        audioRecorder = try AVAudioRecorder(url: url, settings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 48_000,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 128_000,
        ])
        audioRecorder?.prepareToRecord()
        guard audioRecorder?.record() == true else {
            throw NSError(domain: "StereoCapture", code: 11,
                          userInfo: [NSLocalizedDescriptionKey: "The microphone recorder could not start."])
        }
    }

    private func writeManifest(at root: URL) throws {
        let resolution = arSession.currentFrame?.camera.imageResolution ?? .zero
        let payload: [String: Any] = [
            "schema_version": 1,
            "session_id": sessionID,
            "rig_role": role,
            "created_at_utc": ISO8601DateFormatter().string(from: Date()),
            "device": ["model": Self.hardwareModel(), "system_version": UIDevice.current.systemVersion],
            "capture": [
                "target_saved_fps": targetSavedFPS,
                "image_width": Int(resolution.width),
                "image_height": Int(resolution.height),
                "hdr": false,
                "lidar": hasLiDAR,
                "gps": Self.locationAuthorized(locationManager.authorizationStatus),
                "saved_frames": savedFrames,
                "dropped_frames": droppedFrames,
                "write_failures": writeFailures,
                "manual_profile": [
                    "exposure_seconds": requestedExposureSeconds,
                    "iso": requestedISO,
                    "lens_position": requestedLensPosition,
                    "white_balance_temperature_k": requestedWhiteBalanceTemperature,
                ],
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: root.appendingPathComponent("manifest.json"), options: .atomic)
    }

    private static func hardwareModel() -> String {
        var systemInfo = utsname(); uname(&systemInfo)
        return withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
    }

    private func fail(_ error: Error) {
        if isRecording || writer != nil {
            isRecording = false
            if let summary = try? teardownCapture() {
                savedFrames = summary.savedFrames
                writeFailures = summary.failures
            }
        }
        lastError = error
        statusMessage = error.localizedDescription
        isRecording = false
    }

    private static let framesHeader = "frame_id,timestamp_s,arkit_timestamp_s,unix_time_s,image_path,depth_path,confidence_path,depth_width,depth_height,depth_bytes_per_row,confidence_width,confidence_height,confidence_bytes_per_row,fx,fy,cx,cy,image_width,image_height,tracking_state,exposure_seconds,iso,lens_position,wb_red_gain,wb_green_gain,wb_blue_gain,pose_m00,pose_m01,pose_m02,pose_m03,pose_m10,pose_m11,pose_m12,pose_m13,pose_m20,pose_m21,pose_m22,pose_m23,pose_m30,pose_m31,pose_m32,pose_m33"
    private static let locationHeader = "unix_time_s,latitude,longitude,altitude_m,horizontal_accuracy_m,vertical_accuracy_m,speed_mps,course_deg"
    private static let headingHeader = "unix_time_s,magnetic_heading_deg,true_heading_deg,heading_accuracy_deg,x,y,z"
}

extension SessionRecorder: ARSessionDelegate {
    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        Task<Void, Never> { @MainActor [weak self] in
            guard let self else { return }
            self.trackingState = Self.trackingDescription(frame.camera.trackingState)
            guard self.isRecording, frame.timestamp - self.lastSavedTimestamp >= self.saveInterval,
                  let writer = self.writer else { return }
            self.lastSavedTimestamp = frame.timestamp
            let id = self.frameID
            let unixTime = Date().timeIntervalSince1970
            let audioTime = self.audioRecorder?.currentTime ?? 0
            let intrinsics = frame.camera.intrinsics
            let pose = frame.camera.transform
            let tracking = Self.trackingDescription(frame.camera.trackingState)
            let image = frame.capturedImage
            let depth = frame.sceneDepth?.depthMap
            let confidence = frame.sceneDepth?.confidenceMap
            let size = frame.camera.imageResolution
            let exposureSeconds = self.captureDevice?.exposureDuration.seconds ?? .nan
            let iso = self.captureDevice?.iso ?? .nan
            let lensPosition = self.captureDevice?.lensPosition ?? .nan
            let whiteBalance = self.captureDevice?.deviceWhiteBalanceGains
            let depthShape = depth.map { [CVPixelBufferGetWidth($0), CVPixelBufferGetHeight($0), CVPixelBufferGetBytesPerRow($0)] } ?? [0, 0, 0]
            let confidenceShape = confidence.map { [CVPixelBufferGetWidth($0), CVPixelBufferGetHeight($0), CVPixelBufferGetBytesPerRow($0)] } ?? [0, 0, 0]
            let stem = String(format: "%06d", id)
            let depthRelative = depth == nil ? "" : "depth/\(stem).depth"
            let confidenceRelative = confidence == nil ? "" : "confidence/\(stem).conf"
            let poseValues = (0..<4).flatMap { row in (0..<4).map { column in pose[column][row] } }
            var fields: [String] = ["\(id)", "\(audioTime)", "\(frame.timestamp)", "\(unixTime)", "rgb/\(stem).jpg",
                depthRelative, confidenceRelative,
                "\(depthShape[0])", "\(depthShape[1])", "\(depthShape[2])",
                "\(confidenceShape[0])", "\(confidenceShape[1])", "\(confidenceShape[2])",
                "\(intrinsics[0][0])", "\(intrinsics[1][1])", "\(intrinsics[2][0])", "\(intrinsics[2][1])",
                "\(Int(size.width))", "\(Int(size.height))", tracking,
                "\(exposureSeconds)", "\(iso)", "\(lensPosition)",
                "\(whiteBalance?.redGain ?? .nan)", "\(whiteBalance?.greenGain ?? .nan)", "\(whiteBalance?.blueGain ?? .nan)"]
            fields.append(contentsOf: poseValues.map { String($0) })
            let job = FrameWriteJob(id: id, csvLine: fields.joined(separator: ","), image: image,
                                    depth: depth, confidence: confidence)
            let accepted = writer.enqueue(job) { [weak self] summary in
                Task<Void, Never> { @MainActor in
                    guard let self else { return }
                    self.savedFrames = summary.savedFrames
                    self.writeFailures = summary.failures
                }
            }
            if accepted {
                self.frameID += 1
            } else {
                self.droppedFrames += 1
            }
        }
    }

    private static func trackingDescription(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "normal"
        case .notAvailable: return "unavailable"
        case .limited(let reason): return "limited_\(String(describing: reason))"
        }
    }

}

extension SessionRecorder: CLLocationManagerDelegate {
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let value = locations.last else { return }
        Task { @MainActor [weak self] in
            guard let self else { return }
            guard isRecording else { return }
            locationStatus = String(format: "±%.0f m", value.horizontalAccuracy)
            writer?.enqueueLocation([value.timestamp.timeIntervalSince1970, value.coordinate.latitude, value.coordinate.longitude,
                value.altitude, value.horizontalAccuracy, value.verticalAccuracy, value.speed, value.course]
                .map { String($0) }.joined(separator: ","))
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateHeading value: CLHeading) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            guard isRecording else { return }
            writer?.enqueueHeading([Date().timeIntervalSince1970, value.magneticHeading, value.trueHeading, value.headingAccuracy,
                value.x, value.y, value.z].map { String($0) }.joined(separator: ","))
        }
    }
}

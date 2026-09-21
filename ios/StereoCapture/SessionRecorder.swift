import ARKit
import AVFoundation
import CoreImage
import CoreLocation
import SwiftUI
import UIKit

@MainActor
final class SessionRecorder: NSObject, ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var savedFrames = 0
    @Published private(set) var trackingState = "not started"
    @Published private(set) var locationStatus = "waiting"
    @Published private(set) var hasLiDAR = false
    @Published private(set) var statusMessage = "Enter the same session ID on both phones."
    @Published private(set) var lastError: Error?
    @Published private(set) var completedSessionURL: URL?

    private let arSession = ARSession()
    var previewSession: ARSession { arSession }
    private let locationManager = CLLocationManager()
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private var sessionRoot: URL?
    private var framesFile: FileHandle?
    private var locationsFile: FileHandle?
    private var headingsFile: FileHandle?
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
            self.sessionID = sessionID.trimmingCharacters(in: .whitespacesAndNewlines)
            self.role = role
            self.targetSavedFPS = [5, 10, 15].contains(savedFPS) ? savedFPS : 10
            self.saveInterval = 1.0 / Double(self.targetSavedFPS)
            let root = try makeSessionDirectory(sessionID: self.sessionID, role: role)
            sessionRoot = root
            framesFile = try makeCSV(root.appendingPathComponent("frames.csv"), header: Self.framesHeader)
            locationsFile = try makeCSV(root.appendingPathComponent("location.csv"), header: Self.locationHeader)
            headingsFile = try makeCSV(root.appendingPathComponent("heading.csv"), header: Self.headingHeader)
            try startAudio(at: root.appendingPathComponent("audio.m4a"))

            frameID = 0
            savedFrames = 0
            lastSavedTimestamp = -.infinity
            completedSessionURL = nil
            lastError = nil
            isRecording = true
            locationManager.startUpdatingLocation()
            locationManager.startUpdatingHeading()

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
        arSession.pause()
        locationManager.stopUpdatingLocation()
        locationManager.stopUpdatingHeading()
        audioRecorder?.stop()
        audioRecorder = nil
        framesFile?.closeFile(); framesFile = nil
        locationsFile?.closeFile(); locationsFile = nil
        headingsFile?.closeFile(); headingsFile = nil
        guard let root = sessionRoot else { return }
        do {
            try writeManifest(at: root)
            completedSessionURL = root
            statusMessage = "Saved \(savedFrames) frames. Export the session before deleting the app."
        } catch { fail(error) }
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

    private func makeCSV(_ url: URL, header: String) throws -> FileHandle {
        FileManager.default.createFile(atPath: url.path, contents: Data((header + "\n").utf8))
        return try FileHandle(forWritingTo: url)
    }

    private func append(_ line: String, to handle: FileHandle?) {
        try? handle?.write(contentsOf: Data((line + "\n").utf8))
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
        audioRecorder?.record()
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
        Task { @MainActor in
            trackingState = Self.trackingDescription(frame.camera.trackingState)
            guard isRecording, frame.timestamp - lastSavedTimestamp >= saveInterval,
                  let root = sessionRoot else { return }
            lastSavedTimestamp = frame.timestamp
            let id = frameID
            frameID += 1
            savedFrames = frameID
            let unixTime = Date().timeIntervalSince1970
            let audioTime = audioRecorder?.currentTime ?? 0
            let intrinsics = frame.camera.intrinsics
            let pose = frame.camera.transform
            let tracking = Self.trackingDescription(frame.camera.trackingState)
            let image = frame.capturedImage
            let depth = frame.sceneDepth?.depthMap
            let confidence = frame.sceneDepth?.confidenceMap
            let size = frame.camera.imageResolution
            let exposureSeconds = captureDevice?.exposureDuration.seconds ?? .nan
            let iso = captureDevice?.iso ?? .nan
            let lensPosition = captureDevice?.lensPosition ?? .nan
            let whiteBalance = captureDevice?.deviceWhiteBalanceGains
            writeFrame(id: id, timestamp: audioTime, arkitTimestamp: frame.timestamp,
                unixTime: unixTime, image: image,
                depth: depth, confidence: confidence, intrinsics: intrinsics, pose: pose,
                size: size, tracking: tracking, exposureSeconds: exposureSeconds, iso: iso,
                lensPosition: lensPosition, whiteBalance: whiteBalance, root: root)
        }
    }

    private static func trackingDescription(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "normal"
        case .notAvailable: return "unavailable"
        case .limited(let reason): return "limited_\(String(describing: reason))"
        }
    }

    private func writeFrame(id: Int, timestamp: Double, arkitTimestamp: Double, unixTime: Double,
        image: CVPixelBuffer, depth: CVPixelBuffer?, confidence: CVPixelBuffer?,
        intrinsics: simd_float3x3, pose: simd_float4x4, size: CGSize, tracking: String,
        exposureSeconds: Double, iso: Float, lensPosition: Float,
        whiteBalance: AVCaptureDevice.WhiteBalanceGains?, root: URL) {
        let stem = String(format: "%06d", id)
        let imageRelative = "rgb/\(stem).jpg"
        let imageURL = root.appendingPathComponent(imageRelative)
        if let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
           let jpeg = ciContext.jpegRepresentation(of: CIImage(cvPixelBuffer: image), colorSpace: colorSpace, options: [:]) {
            try? jpeg.write(to: imageURL, options: Data.WritingOptions.atomic)
        }
        let depthRelative = depth == nil ? "" : "depth/\(stem).depth"
        let confidenceRelative = confidence == nil ? "" : "confidence/\(stem).conf"
        if let depth { Self.writePixelBuffer(depth, to: root.appendingPathComponent(depthRelative)) }
        if let confidence { Self.writePixelBuffer(confidence, to: root.appendingPathComponent(confidenceRelative)) }

        let depthShape = depth.map { [CVPixelBufferGetWidth($0), CVPixelBufferGetHeight($0), CVPixelBufferGetBytesPerRow($0)] } ?? [0, 0, 0]
        let confidenceShape = confidence.map { [CVPixelBufferGetWidth($0), CVPixelBufferGetHeight($0), CVPixelBufferGetBytesPerRow($0)] } ?? [0, 0, 0]

        let p = pose
        let poseValues = (0..<4).flatMap { row in (0..<4).map { column in p[column][row] } }
        var fields: [String] = ["\(id)", "\(timestamp)", "\(arkitTimestamp)", "\(unixTime)", imageRelative, depthRelative, confidenceRelative,
            "\(depthShape[0])", "\(depthShape[1])", "\(depthShape[2])",
            "\(confidenceShape[0])", "\(confidenceShape[1])", "\(confidenceShape[2])",
            "\(intrinsics[0][0])", "\(intrinsics[1][1])", "\(intrinsics[2][0])", "\(intrinsics[2][1])",
            "\(Int(size.width))", "\(Int(size.height))", tracking,
            "\(exposureSeconds)", "\(iso)", "\(lensPosition)",
            "\(whiteBalance?.redGain ?? .nan)", "\(whiteBalance?.greenGain ?? .nan)", "\(whiteBalance?.blueGain ?? .nan)"]
        fields.append(contentsOf: poseValues.map { String($0) })
        append(fields.joined(separator: ","), to: framesFile)
    }

    nonisolated private static func writePixelBuffer(_ buffer: CVPixelBuffer, to url: URL) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let data = Data(bytes: base, count: CVPixelBufferGetDataSize(buffer))
        try? data.write(to: url, options: .atomic)
    }
}

extension SessionRecorder: CLLocationManagerDelegate {
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let value = locations.last else { return }
        Task { @MainActor [weak self] in
            guard let self else { return }
            guard isRecording else { return }
            locationStatus = String(format: "±%.0f m", value.horizontalAccuracy)
            append([value.timestamp.timeIntervalSince1970, value.coordinate.latitude, value.coordinate.longitude,
                value.altitude, value.horizontalAccuracy, value.verticalAccuracy, value.speed, value.course]
                .map { String($0) }.joined(separator: ","), to: locationsFile)
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateHeading value: CLHeading) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            guard isRecording else { return }
            append([Date().timeIntervalSince1970, value.magneticHeading, value.trueHeading, value.headingAccuracy,
                value.x, value.y, value.z].map { String($0) }.joined(separator: ","), to: headingsFile)
        }
    }
}

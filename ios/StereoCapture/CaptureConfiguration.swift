import Foundation

struct CaptureConfiguration: Equatable {
    var sessionID: String
    var role: String
    var savedFPS: Int

    var normalizedSessionID: String {
        sessionID.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var canStartRecording: Bool { !normalizedSessionID.isEmpty }
    var cameraLabel: String { "CAMERA \(role)" }
    var compactSummary: String { "\(role) · \(savedFPS) FPS" }
}

struct CameraLockProfile: Equatable {
    var uniqueID: String
    var localizedName: String
    var deviceType: String
    var position: String
    var formatWidth: Int
    var formatHeight: Int
    var mediaSubtype: String
    var zoomFactor: Double

    var isCalibrationSafe: Bool {
        position == "back" && deviceType == "wide-angle" && abs(zoomFactor - 1.0) <= 0.001
    }

    func mismatchDescription(comparedTo locked: CameraLockProfile) -> String? {
        guard uniqueID == locked.uniqueID,
              deviceType == locked.deviceType,
              position == locked.position else {
            return "The physical camera changed during recording."
        }
        guard formatWidth == locked.formatWidth,
              formatHeight == locked.formatHeight,
              mediaSubtype == locked.mediaSubtype else {
            return "The camera format changed during recording."
        }
        guard abs(zoomFactor - locked.zoomFactor) <= 0.001 else {
            return "The camera zoom changed during recording."
        }
        return nil
    }

    var manifestPayload: [String: Any] {
        [
            "unique_id": uniqueID,
            "localized_name": localizedName,
            "device_type": deviceType,
            "position": position,
            "format_width": formatWidth,
            "format_height": formatHeight,
            "media_subtype": mediaSubtype,
            "zoom_factor": zoomFactor,
        ]
    }
}

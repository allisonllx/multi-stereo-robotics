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

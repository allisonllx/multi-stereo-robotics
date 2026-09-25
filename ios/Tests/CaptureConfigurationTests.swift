import Foundation

@main
enum CaptureConfigurationTests {
    static func main() {
        var failures = 0

        func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
            if !condition() {
                failures += 1
                FileHandle.standardError.write(Data("FAIL: \(message)\n".utf8))
            }
        }

        let empty = CaptureConfiguration(sessionID: "   ", role: "A", savedFPS: 10)
        expect(!empty.canStartRecording, "whitespace-only session IDs must not permit recording")

        let configured = CaptureConfiguration(sessionID: "  courtyard-loop  ", role: "B", savedFPS: 15)
        expect(configured.normalizedSessionID == "courtyard-loop", "session IDs must be trimmed before capture")
        expect(configured.canStartRecording, "a configured session must permit recording")
        expect(configured.cameraLabel == "CAMERA B", "the camera role must be visible on the capture screen")
        expect(configured.compactSummary == "B · 15 FPS", "the compact settings summary must reflect the active configuration")

        if failures > 0 { exit(1) }
        print("CaptureConfigurationTests passed")
    }
}

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

        let lockedCamera = CameraLockProfile(
            uniqueID: "rear-wide-1",
            localizedName: "Back Camera",
            deviceType: "wide-angle",
            position: "back",
            formatWidth: 1920,
            formatHeight: 1080,
            mediaSubtype: "420f",
            zoomFactor: 1.0
        )
        expect(lockedCamera.isCalibrationSafe,
               "a rear wide-angle camera at 1x must be accepted for calibration capture")
        expect(lockedCamera.mismatchDescription(comparedTo: lockedCamera) == nil,
               "an unchanged camera profile must remain valid")

        var switchedCamera = lockedCamera
        switchedCamera.uniqueID = "rear-telephoto-1"
        expect(switchedCamera.mismatchDescription(comparedTo: lockedCamera) != nil,
               "a physical-camera switch must invalidate the recording")

        var changedFormat = lockedCamera
        changedFormat.formatWidth = 1280
        expect(changedFormat.mismatchDescription(comparedTo: lockedCamera) != nil,
               "a format change must invalidate the recording")

        var changedZoom = lockedCamera
        changedZoom.zoomFactor = 1.1
        expect(changedZoom.mismatchDescription(comparedTo: lockedCamera) != nil,
               "a zoom change must invalidate the recording")

        var virtualCamera = lockedCamera
        virtualCamera.deviceType = "triple-camera"
        expect(!virtualCamera.isCalibrationSafe,
               "a virtual multi-camera device must not pass the physical wide-angle check")

        if failures > 0 { exit(1) }
        print("CaptureConfigurationTests passed")
    }
}

import SwiftUI

struct ContentView: View {
    @StateObject private var recorder = SessionRecorder()
    @State private var sessionID = ""
    @State private var role = "A"
    @State private var savedFPS = 10

    var body: some View {
        VStack(spacing: 18) {
            Text("Stereo Capture").font(.largeTitle.bold())
            CameraPreview(session: recorder.previewSession)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(alignment: .topLeading) {
                    Text(role == "A" ? "CAMERA A" : "CAMERA B")
                        .font(.caption.bold()).padding(8).background(.black.opacity(0.6)).foregroundStyle(.white)
                }
                .frame(maxHeight: .infinity)
            HStack {
                TextField("Shared session ID", text: $sessionID)
                    .textFieldStyle(.roundedBorder)
                Picker("Rig role", selection: $role) {
                    Text("Camera A").tag("A")
                    Text("Camera B").tag("B")
                }.pickerStyle(.segmented)
                Picker("Saved FPS", selection: $savedFPS) {
                    Text("5 fps").tag(5)
                    Text("10 fps").tag(10)
                    Text("15 fps").tag(15)
                }.pickerStyle(.segmented)
            }.disabled(recorder.isRecording)

            Text("Matched manual profile: 1/120 s · ISO 100 · 5000 K · focus 0.75")
                .font(.caption).foregroundStyle(.secondary)

            HStack(spacing: 28) {
                StatusCell(label: "Saved", value: "\(recorder.savedFrames)")
                StatusCell(label: "Target", value: "\(savedFPS) fps")
                StatusCell(label: "Tracking", value: recorder.trackingState)
                StatusCell(label: "LiDAR", value: recorder.hasLiDAR ? "active" : "unavailable")
                StatusCell(label: "GPS", value: recorder.locationStatus)
            }

            Text(recorder.statusMessage)
                .foregroundStyle(recorder.lastError == nil ? Color.secondary : Color.red)

            Button(recorder.isRecording ? "Stop and Save" : "Start Recording") {
                if recorder.isRecording {
                    recorder.stop()
                } else {
                    recorder.start(sessionID: sessionID, role: role, savedFPS: savedFPS)
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(recorder.isRecording ? .red : .blue)
            .disabled(!recorder.isRecording && sessionID.trimmingCharacters(in: .whitespaces).isEmpty)

            if let exportURL = recorder.completedSessionURL {
                ShareLink(item: exportURL) { Label("Export Session", systemImage: "square.and.arrow.up") }
            }
        }
        .padding(30)
        .task { recorder.prepare() }
    }
}

private struct StatusCell: View {
    let label: String
    let value: String
    var body: some View {
        VStack { Text(value).font(.headline); Text(label).font(.caption).foregroundStyle(.secondary) }
    }
}

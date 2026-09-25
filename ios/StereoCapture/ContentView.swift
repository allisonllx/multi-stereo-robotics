import SwiftUI

struct ContentView: View {
    @StateObject private var recorder = SessionRecorder()
    @AppStorage("capture.sessionID") private var sessionID = ""
    @AppStorage("capture.role") private var role = "A"
    @AppStorage("capture.savedFPS") private var savedFPS = 10
    @State private var showsQuickSettings = false
    @State private var showsFullSettings = false
    @State private var recordingStartedAt: Date?

    private var configuration: CaptureConfiguration {
        CaptureConfiguration(sessionID: sessionID, role: role, savedFPS: savedFPS)
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: recorder.previewSession).ignoresSafeArea()
            LinearGradient(
                colors: [.black.opacity(0.72), .clear, .clear, .black.opacity(0.82)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
            .allowsHitTesting(false)

            VStack(spacing: 0) {
                captureHeader
                Spacer(minLength: 24)

                if showsQuickSettings && !recorder.isRecording {
                    QuickSettingsPanel(
                        role: $role,
                        savedFPS: $savedFPS,
                        showAdvanced: { showsFullSettings = true }
                    )
                    .padding(.horizontal, 16)
                    .padding(.bottom, 14)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                recordingReadout.padding(.bottom, 12)
                captureControls
            }
        }
        .preferredColorScheme(.dark)
        .animation(.snappy(duration: 0.28), value: showsQuickSettings)
        .sheet(isPresented: $showsFullSettings) {
            CaptureSettingsView(sessionID: $sessionID, role: $role, savedFPS: $savedFPS)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .task { recorder.prepare() }
    }

    private var captureHeader: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(configuration.cameraLabel)
                    .font(.caption.weight(.bold))
                    .tracking(1.4)
                Text(configuration.normalizedSessionID.isEmpty ? "SESSION NOT SET" : configuration.normalizedSessionID)
                    .font(.caption2.monospaced())
                    .lineLimit(1)
                    .foregroundStyle(configuration.canStartRecording ? .white.opacity(0.72) : .yellow)
            }
            Spacer()
            SensorPill(icon: "location.fill", active: recorder.locationStatus != "disabled", label: "GPS")
            SensorPill(icon: "viewfinder", active: recorder.trackingState != "not started", label: "AR")
            SensorPill(icon: "square.3.layers.3d", active: recorder.hasLiDAR, label: "LiDAR")
            Button {
                guard !recorder.isRecording else { return }
                showsQuickSettings.toggle()
            } label: {
                Image(systemName: showsQuickSettings ? "chevron.down" : "slider.horizontal.3")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(width: 38, height: 38)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .accessibilityLabel(showsQuickSettings ? "Close quick settings" : "Open quick settings")
            .disabled(recorder.isRecording)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 16)
        .padding(.top, 10)
    }

    @ViewBuilder
    private var recordingReadout: some View {
        if recorder.isRecording {
            VStack(spacing: 7) {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    HStack(spacing: 7) {
                        Circle().fill(.red).frame(width: 7, height: 7)
                        Text(elapsedTime(at: context.date))
                            .font(.system(.body, design: .monospaced).weight(.semibold))
                    }
                }
                Text("\(recorder.savedFrames) saved  ·  \(recorder.droppedFrames) dropped")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(recorder.writeFailures == 0 ? .white.opacity(0.72) : .red)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .background(.black.opacity(0.48), in: Capsule())
        } else if recorder.lastError != nil {
            Text(recorder.statusMessage)
                .font(.caption)
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(.red.opacity(0.8), in: Capsule())
                .lineLimit(2)
        }
    }

    private var captureControls: some View {
        HStack(alignment: .center) {
            Group {
                if let exportURL = recorder.completedSessionURL, !recorder.isRecording {
                    ShareLink(item: exportURL) {
                        ControlIcon(symbol: "square.and.arrow.up", label: "Export")
                    }
                } else {
                    ControlIcon(symbol: "photo.on.rectangle", label: "No session").opacity(0.42)
                }
            }
            .frame(maxWidth: .infinity)

            Button(action: toggleRecording) {
                ZStack {
                    Circle().stroke(.white, lineWidth: 4).frame(width: 76, height: 76)
                    RoundedRectangle(cornerRadius: recorder.isRecording ? 7 : 32)
                        .fill(.red)
                        .frame(width: recorder.isRecording ? 31 : 62, height: recorder.isRecording ? 31 : 62)
                }
                .contentShape(Circle())
            }
            .disabled(!recorder.isRecording && !configuration.canStartRecording)
            .opacity(!recorder.isRecording && !configuration.canStartRecording ? 0.45 : 1)
            .accessibilityLabel(recorder.isRecording ? "Stop and save recording" : "Start recording")

            Button {
                guard !recorder.isRecording else { return }
                showsFullSettings = true
            } label: {
                ControlIcon(symbol: "gearshape.fill", label: configuration.compactSummary)
            }
            .disabled(recorder.isRecording)
            .frame(maxWidth: .infinity)
        }
        .padding(.horizontal, 14)
        .padding(.top, 14)
        .padding(.bottom, 10)
        .background(.black.opacity(0.68))
    }

    private func toggleRecording() {
        if recorder.isRecording {
            recorder.stop()
            recordingStartedAt = nil
        } else {
            showsQuickSettings = false
            recordingStartedAt = .now
            recorder.start(
                sessionID: configuration.normalizedSessionID,
                role: configuration.role,
                savedFPS: configuration.savedFPS
            )
            if !recorder.isRecording { recordingStartedAt = nil }
        }
    }

    private func elapsedTime(at date: Date) -> String {
        let elapsed = max(0, Int(date.timeIntervalSince(recordingStartedAt ?? date)))
        return String(format: "%02d:%02d", elapsed / 60, elapsed % 60)
    }
}

private struct QuickSettingsPanel: View {
    @Binding var role: String
    @Binding var savedFPS: Int
    let showAdvanced: () -> Void

    var body: some View {
        VStack(spacing: 14) {
            HStack {
                Label("Capture settings", systemImage: "slider.horizontal.3").font(.headline)
                Spacer()
                Button("Advanced", action: showAdvanced).font(.subheadline.weight(.semibold))
            }
            Picker("Camera", selection: $role) {
                Text("Camera A").tag("A")
                Text("Camera B").tag("B")
            }
            .pickerStyle(.segmented)
            Picker("Saved frame rate", selection: $savedFPS) {
                Text("5 FPS").tag(5)
                Text("10 FPS").tag(10)
                Text("15 FPS").tag(15)
            }
            .pickerStyle(.segmented)
            HStack(spacing: 0) {
                ProfileValue(value: "1/120", label: "SHUTTER")
                ProfileValue(value: "100", label: "ISO")
                ProfileValue(value: "5000 K", label: "WHITE BALANCE")
                ProfileValue(value: "0.75", label: "FOCUS")
            }
        }
        .padding(16)
        .foregroundStyle(.white)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
    }
}

private struct CaptureSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var sessionID: String
    @Binding var role: String
    @Binding var savedFPS: Int

    var body: some View {
        NavigationStack {
            Form {
                Section("Stereo session") {
                    TextField("Shared session ID", text: $sessionID)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Picker("Rig camera", selection: $role) {
                        Text("Camera A").tag("A")
                        Text("Camera B").tag("B")
                    }
                    Picker("Saved frame rate", selection: $savedFPS) {
                        Text("5 FPS").tag(5)
                        Text("10 FPS").tag(10)
                        Text("15 FPS").tag(15)
                    }
                }
                Section {
                    LabeledContent("Shutter", value: "1/120 s")
                    LabeledContent("ISO", value: "100")
                    LabeledContent("White balance", value: "5000 K")
                    LabeledContent("Focus", value: "0.75")
                } header: {
                    Text("Matched camera profile")
                } footer: {
                    Text("These values are locked by the recorder so both phones use the same capture profile.")
                }
                Section("Sensors") {
                    Label("ARKit pose and tracking", systemImage: "viewfinder")
                    Label("GPS and compass when permitted", systemImage: "location.fill")
                    Label("Scene depth on LiDAR devices", systemImage: "square.3.layers.3d")
                }
            }
            .navigationTitle("Capture Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }.fontWeight(.semibold)
                }
            }
        }
    }
}

private struct SensorPill: View {
    let icon: String
    let active: Bool
    let label: String

    var body: some View {
        Image(systemName: icon)
            .font(.caption2.weight(.bold))
            .foregroundStyle(active ? .green : .white.opacity(0.48))
            .frame(width: 26, height: 26)
            .background(.black.opacity(0.42), in: Circle())
            .accessibilityLabel("\(label) \(active ? "active" : "inactive")")
    }
}

private struct ProfileValue: View {
    let value: String
    let label: String

    var body: some View {
        VStack(spacing: 3) {
            Text(value).font(.caption.weight(.semibold)).lineLimit(1)
            Text(label).font(.system(size: 8, weight: .bold)).foregroundStyle(.secondary).lineLimit(1)
        }
        .frame(maxWidth: .infinity)
    }
}

private struct ControlIcon: View {
    let symbol: String
    let label: String

    var body: some View {
        VStack(spacing: 5) {
            Image(systemName: symbol).font(.system(size: 20, weight: .medium))
            Text(label).font(.system(size: 10, weight: .medium)).lineLimit(1)
        }
        .foregroundStyle(.white)
    }
}

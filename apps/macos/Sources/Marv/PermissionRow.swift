import MarvIPC
import SwiftUI

struct PermissionRow: View {
    let capability: Capability
    let status: Bool
    let compact: Bool
    let action: () -> Void

    init(capability: Capability, status: Bool, compact: Bool = false, action: @escaping () -> Void) {
        self.capability = capability
        self.status = status
        self.compact = compact
        self.action = action
    }

    var body: some View {
        HStack(spacing: self.compact ? 10 : 12) {
            ZStack {
                Circle().fill(self.status ? Color.green.opacity(0.2) : Color.gray.opacity(0.15))
                    .frame(width: self.iconSize, height: self.iconSize)
                Image(systemName: self.icon)
                    .foregroundStyle(self.status ? Color.green : Color.secondary)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(self.title).font(.body.weight(.semibold))
                Text(self.subtitle).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if self.status {
                Label("Granted", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            } else {
                Button("Grant") { self.action() }
                    .buttonStyle(.bordered)
            }
        }
        .padding(.vertical, self.compact ? 4 : 6)
    }

    private var iconSize: CGFloat {
        self.compact ? 28 : 32
    }

    private var title: String {
        switch self.capability {
        case .appleScript: "Automation (AppleScript)"
        case .notifications: "Notifications"
        case .accessibility: "Accessibility"
        case .screenRecording: "Screen Recording"
        case .microphone: "Microphone"
        case .speechRecognition: "Speech Recognition"
        case .camera: "Camera"
        case .location: "Location"
        }
    }

    private var subtitle: String {
        switch self.capability {
        case .appleScript:
            "Control other apps (e.g. Terminal) for automation actions"
        case .notifications: "Show desktop alerts for agent activity"
        case .accessibility: "Control UI elements when an action requires it"
        case .screenRecording: "Capture the screen for context or screenshots"
        case .microphone: "Allow Voice Wake and audio capture"
        case .speechRecognition: "Transcribe Voice Wake trigger phrases on-device"
        case .camera: "Capture photos and video from the camera"
        case .location: "Share location when requested by the agent"
        }
    }

    private var icon: String {
        switch self.capability {
        case .appleScript: "applescript"
        case .notifications: "bell"
        case .accessibility: "hand.raised"
        case .screenRecording: "display"
        case .microphone: "mic"
        case .speechRecognition: "waveform"
        case .camera: "camera"
        case .location: "location"
        }
    }
}

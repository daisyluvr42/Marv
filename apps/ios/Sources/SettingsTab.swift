import SwiftUI

struct SettingsTab: View {
    @Environment(CompanionAppModel.self) private var appModel

    var body: some View {
        @Bindable var appModel = self.appModel
        NavigationStack {
            Form {
                Section("Gateway") {
                    TextField("ws://gateway-host:18789", text: $appModel.gatewayURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    SecureField("Shared token (recommended)", text: $appModel.gatewayToken)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField("Shared password (optional)", text: $appModel.gatewayPassword)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }

                Section("Companion Mode") {
                    TextField("Chat session key", text: $appModel.sessionKey)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Toggle("Enable agent camera snapshots", isOn: $appModel.cameraNodeEnabled)
                    Text("When enabled, the app opens a separate node connection that only declares `camera.list` and `camera.snap`.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Text("The camera node still needs deliberate gateway allowlisting (`gateway.nodes.allowCommands: [\"camera.snap\"]`) and usually a second pairing approval for the `node` role.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Status") {
                    KeyValueRow(label: "Operator", value: self.appModel.operatorStatusText)
                    KeyValueRow(label: "Camera Node", value: self.appModel.nodeStatusText)
                }

                Section {
                    Button("Save And Connect") {
                        Task {
                            await appModel.saveAndReconnect()
                        }
                    }
                    .disabled(appModel.isConnecting)

                    Button("Refresh Dashboard") {
                        Task {
                            await appModel.refreshDashboard()
                        }
                    }
                    .disabled(!appModel.isOperatorConnected)

                    Button("Disconnect", role: .destructive) {
                        Task {
                            await appModel.disconnect()
                        }
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

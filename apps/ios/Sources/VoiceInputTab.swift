import SwiftUI

struct VoiceInputTab: View {
    @Environment(CompanionAppModel.self) private var appModel
    @Environment(SpeechInputModel.self) private var speechInput
    @Environment(\.verticalSizeClass) private var verticalSizeClass

    var body: some View {
        @Bindable var appModel = self.appModel
        @Bindable var speechInput = self.speechInput
        NavigationStack {
            Form {
                Section("Session") {
                    TextField("Session key", text: $appModel.sessionKey)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Text("Voice input is transcribed locally, then sent as a normal chat message.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Transcript") {
                    TextEditor(text: $speechInput.transcript)
                        .frame(minHeight: self.verticalSizeClass == .compact ? 120 : 160)
                    Text(speechInput.authorizationText)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    if let errorText = speechInput.errorText, !errorText.isEmpty {
                        Text(errorText)
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }
                }

                Section {
                    Button(speechInput.isListening ? "Stop Listening" : "Start Listening") {
                        speechInput.toggleListening()
                    }
                    .disabled(!appModel.isOperatorConnected && !speechInput.isListening)

                    Button("Send To Agent") {
                        Task {
                            await appModel.sendTranscript()
                        }
                    }
                    .disabled(!appModel.isOperatorConnected || speechInput.transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    Button("Reset Transcript", role: .destructive) {
                        speechInput.reset()
                    }
                }
            }
            .navigationTitle("Voice Input")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

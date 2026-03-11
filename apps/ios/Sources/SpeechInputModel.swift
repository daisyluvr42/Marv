import AVFoundation
import Foundation
import Observation
import Speech

@MainActor
@Observable
final class SpeechInputModel {
    var transcript = ""
    var isListening = false
    var errorText: String?
    var authorizationText = "Mic + speech permissions needed"

    @ObservationIgnored
    private let audioEngine = AVAudioEngine()
    @ObservationIgnored
    private let recognizer = SFSpeechRecognizer()
    @ObservationIgnored
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    @ObservationIgnored
    private var recognitionTask: SFSpeechRecognitionTask?

    func toggleListening() {
        if self.isListening {
            self.stopListening()
        } else {
            Task {
                await self.startListening()
            }
        }
    }

    func reset() {
        self.stopListening()
        self.transcript = ""
        self.errorText = nil
    }

    private func startListening() async {
        self.errorText = nil

        guard await self.requestPermissions() else {
            return
        }
        guard let recognizer = self.recognizer, recognizer.isAvailable else {
            self.errorText = "Speech recognizer unavailable"
            return
        }

        self.stopListening()

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if #available(iOS 13, *) {
            request.requiresOnDeviceRecognition = false
        }
        self.recognitionRequest = request

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            self.errorText = error.localizedDescription
            return
        }

        let inputNode = self.audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }

        self.audioEngine.prepare()
        do {
            try self.audioEngine.start()
        } catch {
            self.errorText = error.localizedDescription
            return
        }

        self.isListening = true
        self.authorizationText = "Listening"
        self.recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            Task { @MainActor in
                if let result {
                    self.transcript = result.bestTranscription.formattedString
                    if result.isFinal {
                        self.stopListening()
                    }
                }
                if let error {
                    self.errorText = error.localizedDescription
                    self.stopListening()
                }
            }
        }
    }

    func stopListening() {
        self.audioEngine.stop()
        self.audioEngine.inputNode.removeTap(onBus: 0)
        self.recognitionRequest?.endAudio()
        self.recognitionTask?.cancel()
        self.recognitionTask = nil
        self.recognitionRequest = nil
        self.isListening = false
        if self.errorText == nil {
            self.authorizationText = "Ready"
        }
        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            // Best effort only.
        }
    }

    private func requestPermissions() async -> Bool {
        let speechStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
        guard speechStatus == .authorized else {
            self.authorizationText = "Speech permission denied"
            self.errorText = "Enable speech recognition in iOS Settings"
            return false
        }

        let microphoneGranted = await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
        guard microphoneGranted else {
            self.authorizationText = "Microphone permission denied"
            self.errorText = "Enable microphone access in iOS Settings"
            return false
        }

        self.authorizationText = "Ready"
        return true
    }
}

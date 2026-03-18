import AppKit
import Foundation
import MarvIPC
import SwiftUI

extension OnboardingView {
    func openSettings(tab: SettingsTab) {
        SettingsTabRouter.request(tab)
        self.openSettings()
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: .marvSelectSettingsTab, object: tab)
        }
    }

    func handleBack() {
        withAnimation {
            self.currentPage = max(0, self.currentPage - 1)
        }
    }

    func handleNext() {
        if self.currentPage < self.pageCount - 1 {
            withAnimation { self.currentPage += 1 }
        } else {
            Task { await self.finishOnboarding() }
        }
    }

    func updateDefaultModel() {
        if self.selectedProvider == "custom" {
            // For custom, pre-fill from customModelId if set
            self.selectedModel = self.customModelId
        } else if let models = Self.providerModels[self.selectedProvider], let first = models.first {
            self.selectedModel = first.id
        } else {
            self.selectedModel = ""
        }
    }

    // MARK: - Config Writing

    /// Save provider, API key, model, and gateway mode to ~/.marv/marv.json and credentials.
    @MainActor
    func finishOnboarding() async {
        guard !self.savingConfig else { return }
        self.savingConfig = true
        defer { self.savingConfig = false }

        // Write API key to credentials directory
        self.writeApiKeyCredential()

        // Write provider + model + gateway config
        await self.saveProviderAndModelConfig()

        // Set gateway mode to local
        self.state.connectionMode = .local

        // Ensure workspace exists
        await self.loadWorkspaceDefaults()
        await self.ensureDefaultWorkspace()

        // Mark onboarding complete
        UserDefaults.standard.set(true, forKey: "marv.onboardingSeen")
        UserDefaults.standard.set(currentOnboardingVersion, forKey: onboardingVersionKey)
        OnboardingController.shared.close()
    }

    /// Write the API key to ~/.marv/credentials/ in the format the CLI expects.
    /// The CLI stores auth profiles in the agent dir (auth-profiles.json), but for
    /// onboarding we write a simple provider credential file that the gateway picks up.
    private func writeApiKeyCredential() {
        let trimmedKey = self.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else { return }

        let stateDir = MarvPaths.stateDirURL
        let credentialsDir = stateDir.appendingPathComponent("credentials", isDirectory: true)

        do {
            try FileManager.default.createDirectory(at: credentialsDir, withIntermediateDirectories: true)
        } catch {
            return
        }

        // Write a provider-specific credential file
        let filename: String
        switch self.selectedProvider {
        case "anthropic":
            filename = "anthropic.json"
        case "openai":
            filename = "openai.json"
        case "google":
            filename = "google.json"
        case "moonshot":
            filename = "moonshot.json"
        case "custom":
            filename = "custom.json"
        default:
            filename = "\(self.selectedProvider).json"
        }

        var credDict: [String: Any] = [
            "type": "api_key",
            "provider": self.selectedProvider,
            "key": trimmedKey,
        ]

        if self.selectedProvider == "custom" {
            let trimmedUrl = self.customBaseUrl.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedUrl.isEmpty {
                credDict["baseUrl"] = trimmedUrl
            }
        }

        let credURL = credentialsDir.appendingPathComponent(filename)
        guard let data = try? JSONSerialization.data(
            withJSONObject: credDict,
            options: [.prettyPrinted, .sortedKeys])
        else {
            return
        }
        try? data.write(to: credURL, options: [.atomic])

        // Set owner-only permissions (0600)
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o600],
            ofItemAtPath: credURL.path)
    }

    /// Save provider and model selection to marv.json config.
    @MainActor
    private func saveProviderAndModelConfig() async {
        var root = await ConfigStore.load()

        // Set gateway mode to local
        var gateway = root["gateway"] as? [String: Any] ?? [:]
        gateway["mode"] = "local"
        root["gateway"] = gateway

        // Set default model pool with the selected model
        var agents = root["agents"] as? [String: Any] ?? [:]
        var defaults = agents["defaults"] as? [String: Any] ?? [:]

        let effectiveModel = self.selectedProvider == "custom"
            ? self.customModelId.trimmingCharacters(in: .whitespacesAndNewlines)
            : self.selectedModel

        if !effectiveModel.isEmpty {
            // Model pool format: list of entries with id + model
            let modelRef: String
            if self.selectedProvider == "custom" {
                let baseUrl = self.customBaseUrl.trimmingCharacters(in: .whitespacesAndNewlines)
                modelRef = baseUrl.isEmpty ? effectiveModel : "\(self.selectedProvider):\(effectiveModel)"
            } else {
                modelRef = "\(self.selectedProvider):\(effectiveModel)"
            }
            defaults["modelPool"] = [
                ["id": "primary", "model": modelRef],
            ]
        }

        if defaults.isEmpty {
            agents.removeValue(forKey: "defaults")
        } else {
            agents["defaults"] = defaults
        }
        if agents.isEmpty {
            root.removeValue(forKey: "agents")
        } else {
            root["agents"] = agents
        }

        do {
            try await ConfigStore.save(root)
        } catch {
            // Fall back to direct file write
            MarvConfigFile.saveDict(root)
        }
    }

    func copyToPasteboard(_ text: String) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
    }
}

import AppKit
import Foundation
import MarvIPC
import SwiftUI

extension OnboardingView {
    private static let authStoreVersion = 1

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
            // Auto-start setup when arriving at page 3
            if self.activePageIndex(for: self.currentPage) == 3, self.setupPhase == .idle {
                Task { await self.runSetup() }
            }
        } else {
            // On the setup page: "Done" closes, "Retry" re-runs setup
            if self.setupPhase == .done {
                self.finishAndClose()
            } else if self.isSetupFailed {
                Task { await self.runSetup() }
            }
        }
    }

    func updateDefaultModel() {
        if self.selectedProvider == "custom" {
            self.selectedModel = self.resolvedCustomModelId
        } else if let models = Self.providerModels[self.selectedProvider], let first = models.first {
            self.selectedModel = first.id
        } else {
            self.selectedModel = ""
        }
    }

    // MARK: - Setup Flow

    /// Run the full setup sequence: save config → check env → install CLI → start gateway.
    @MainActor
    func runSetup() async {
        guard !self.savingConfig else { return }
        self.savingConfig = true
        defer { self.savingConfig = false }

        // Phase 1: Save configuration
        withAnimation { self.setupPhase = .savingConfig }
        self.setupDetailMessage = "Writing API key and config..."
        self.writeApiKeyAuthProfile()
        await self.saveProviderAndModelConfig()
        self.state.connectionMode = .local

        // Phase 2: Check environment
        withAnimation { self.setupPhase = .checkingEnvironment }
        self.setupDetailMessage = "Looking for Node.js..."

        let envStatus = await Task.detached(priority: .userInitiated) {
            GatewayEnvironment.check()
        }.value

        switch envStatus.kind {
        case .missingNode:
            withAnimation {
                self.setupPhase = .failed(
                    "Node.js 22+ is required but was not found. Install Node.js from nodejs.org and try again.")
            }
            self.setupDetailMessage = ""
            return

        case .ok:
            // CLI already installed, skip install step
            self.setupDetailMessage = "Node \(envStatus.nodeVersion ?? "?") found, CLI ready."

        case .missingGateway, .incompatible:
            // Phase 3: Install CLI
            withAnimation { self.setupPhase = .installingCLI }
            self.setupDetailMessage = "Downloading and installing..."

            await CLIInstaller.install { status in
                self.setupDetailMessage = status
            }

            // Re-check after install
            let recheck = await Task.detached(priority: .userInitiated) {
                GatewayEnvironment.check()
            }.value

            if recheck.kind != .ok {
                withAnimation {
                    self.setupPhase = .failed(
                        "CLI installation did not complete successfully. \(recheck.message)")
                }
                self.setupDetailMessage = ""
                return
            }

        case .checking:
            break // shouldn't happen

        case let .error(msg):
            withAnimation {
                self.setupPhase = .failed("Environment check failed: \(msg)")
            }
            self.setupDetailMessage = ""
            return
        }

        // Phase 4: Start gateway (skip if already running)
        withAnimation { self.setupPhase = .startingGateway }
        let port = GatewayEnvironment.gatewayPort()
        self.setupDetailMessage = "Checking for running gateway on port \(port)..."

        let alreadyRunning = await self.waitForGateway(port: port, timeout: 2)
        if alreadyRunning {
            self.setupDetailMessage = "Gateway already running."
        } else {
            self.setupDetailMessage = "Launching gateway on port \(port)..."
            let gateway = GatewayProcessManager.shared
            gateway.setActive(true)

            // Wait for gateway to become reachable (up to 15s)
            let gatewayReady = await self.waitForGateway(port: port, timeout: 15)
            if !gatewayReady {
                withAnimation {
                    self.setupPhase = .failed(
                        "The local gateway did not become reachable on port \(port). Check your Node/CLI install and try again.")
                }
                self.setupDetailMessage = ""
                return
            }
        }

        // Ensure workspace + mark onboarding complete
        await self.loadWorkspaceDefaults()
        await self.ensureDefaultWorkspace()
        UserDefaults.standard.set(true, forKey: "marv.onboardingSeen")
        UserDefaults.standard.set(currentOnboardingVersion, forKey: onboardingVersionKey)
        AppStateStore.shared.onboardingSeen = true

        withAnimation { self.setupPhase = .done }
        self.setupDetailMessage = ""
    }

    /// Close onboarding after successful setup.
    func finishAndClose() {
        OnboardingController.shared.close()
    }

    /// Skip setup and close onboarding (user can set up later from Settings).
    func skipSetup() {
        // Still save config so provider/model/key aren't lost
        self.writeApiKeyAuthProfile()
        Task {
            await self.saveProviderAndModelConfig()
            self.state.connectionMode = .local
            await self.loadWorkspaceDefaults()
            await self.ensureDefaultWorkspace()
            UserDefaults.standard.set(true, forKey: "marv.onboardingSeen")
            UserDefaults.standard.set(currentOnboardingVersion, forKey: onboardingVersionKey)
            AppStateStore.shared.onboardingSeen = true
            OnboardingController.shared.close()
        }
    }

    /// Poll localhost gateway health endpoint until it responds or timeout.
    private func waitForGateway(port: Int, timeout: Int) async -> Bool {
        let url = URL(string: "http://127.0.0.1:\(port)/health")!
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 2
        let session = URLSession(configuration: config)

        let deadline = Date().addingTimeInterval(TimeInterval(timeout))
        while Date() < deadline {
            do {
                let (_, response) = try await session.data(from: url)
                if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                    return true
                }
            } catch {
                // Not ready yet
            }
            try? await Task.sleep(for: .seconds(1))
        }
        return false
    }

    private var resolvedCustomModelId: String {
        let selected = self.selectedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        if !selected.isEmpty {
            return selected
        }
        return self.customModelId.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var configuredProviderId: String {
        if self.selectedProvider != "custom" {
            return self.selectedProvider
        }

        let baseUrl = self.customBaseUrl.trimmingCharacters(in: .whitespacesAndNewlines)
        let providers = (MarvConfigFile.loadDict()["models"] as? [String: Any])?["providers"] as? [String: Any] ?? [:]
        return self.resolveUniqueCustomProviderId(baseUrl: baseUrl, providers: providers)
    }

    private var authProfileId: String {
        "\(self.configuredProviderId):default"
    }

    private func writeApiKeyAuthProfile() {
        let trimmedKey = self.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else { return }

        do {
            try FileManager.default.createDirectory(
                at: MarvPaths.authProfilesURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
        } catch {
            return
        }

        var root = self.loadJSONDictionary(from: MarvPaths.authProfilesURL) ?? [:]
        root["version"] = (root["version"] as? Int) ?? Self.authStoreVersion
        var profiles = root["profiles"] as? [String: Any] ?? [:]
        profiles[self.authProfileId] = [
            "type": "api_key",
            "provider": self.configuredProviderId,
            "key": trimmedKey,
        ]
        root["profiles"] = profiles

        guard let data = try? JSONSerialization.data(
            withJSONObject: root,
            options: [.prettyPrinted, .sortedKeys])
        else {
            return
        }
        try? data.write(to: MarvPaths.authProfilesURL, options: [.atomic])

        // Set owner-only permissions (0600)
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o600],
            ofItemAtPath: MarvPaths.authProfilesURL.path)
    }

    /// Save provider and model selection to marv.json config.
    @MainActor
    private func saveProviderAndModelConfig() async {
        var root = await ConfigStore.load()

        // Set gateway mode to local
        var gateway = root["gateway"] as? [String: Any] ?? [:]
        gateway["mode"] = "local"
        root["gateway"] = gateway

        var auth = root["auth"] as? [String: Any] ?? [:]
        var authProfiles = auth["profiles"] as? [String: Any] ?? [:]
        authProfiles[self.authProfileId] = [
            "provider": self.configuredProviderId,
            "mode": "api_key",
        ]
        auth["profiles"] = authProfiles
        var authOrder = auth["order"] as? [String: Any] ?? [:]
        authOrder[self.configuredProviderId] = [self.authProfileId]
        auth["order"] = authOrder
        root["auth"] = auth

        var agents = root["agents"] as? [String: Any] ?? [:]
        var defaults = agents["defaults"] as? [String: Any] ?? [:]

        let effectiveModel = self.selectedProvider == "custom"
            ? self.resolvedCustomModelId
            : self.selectedModel.trimmingCharacters(in: .whitespacesAndNewlines)

        if !effectiveModel.isEmpty {
            defaults["model"] = [
                "primary": "\(self.configuredProviderId)/\(effectiveModel)",
            ]
        }
        if let modelPool = defaults["modelPool"], !(modelPool is String) {
            defaults.removeValue(forKey: "modelPool")
        }

        var models = root["models"] as? [String: Any] ?? [:]
        var selections = models["selections"] as? [String: Any] ?? [:]
        if !effectiveModel.isEmpty {
            selections[self.configuredProviderId] = ["\(self.configuredProviderId)/\(effectiveModel)"]
        }
        if selections.isEmpty {
            models.removeValue(forKey: "selections")
        } else {
            models["selections"] = selections
        }
        if self.selectedProvider == "custom",
           let providerConfig = self.customProviderConfig(modelId: effectiveModel)
        {
            var providers = models["providers"] as? [String: Any] ?? [:]
            providers[self.configuredProviderId] = providerConfig
            models["providers"] = providers
        }
        if models.isEmpty {
            root.removeValue(forKey: "models")
        } else {
            root["models"] = models
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

    private func customProviderConfig(modelId: String) -> [String: Any]? {
        let baseUrl = self.customBaseUrl.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !baseUrl.isEmpty, !modelId.isEmpty else { return nil }

        let api = self.customApiCompatibility == "anthropic" ? "anthropic-messages" : "openai-completions"
        return [
            "baseUrl": baseUrl,
            "api": api,
            "models": [
                [
                    "id": modelId,
                    "name": modelId,
                ],
            ],
        ]
    }

    private func resolveUniqueCustomProviderId(baseUrl: String, providers: [String: Any]) -> String {
        let requestedId = Self.normalizedCustomProviderId(from: baseUrl)
        guard !requestedId.isEmpty else { return "custom" }

        if let existing = providers[requestedId] as? [String: Any],
           let existingBaseUrl = existing["baseUrl"] as? String,
           existingBaseUrl.trimmingCharacters(in: .whitespacesAndNewlines) != baseUrl
        {
            var suffix = 2
            while true {
                let candidate = "\(requestedId)-\(suffix)"
                if let collision = providers[candidate] as? [String: Any],
                   let collisionBaseUrl = collision["baseUrl"] as? String,
                   collisionBaseUrl.trimmingCharacters(in: .whitespacesAndNewlines) != baseUrl
                {
                    suffix += 1
                    continue
                }
                return candidate
            }
        }

        return requestedId
    }

    private static func normalizedCustomProviderId(from baseUrl: String) -> String {
        guard let url = URL(string: baseUrl), let host = url.host?.lowercased(), !host.isEmpty else {
            return "custom"
        }

        let portSuffix = url.port.map { "-\($0)" } ?? ""
        let raw = "custom-\(host)\(portSuffix)"
        let normalized = raw
            .replacingOccurrences(of: "[^a-z0-9-]+", with: "-", options: .regularExpression)
            .replacingOccurrences(of: "^-+|-+$", with: "", options: .regularExpression)
        return normalized.isEmpty ? "custom" : normalized
    }

    private func loadJSONDictionary(from url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        return object
    }

    func copyToPasteboard(_ text: String) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
    }
}

#if DEBUG
@MainActor
extension OnboardingView {
    func _testWriteApiKeyAuthProfile() {
        self.writeApiKeyAuthProfile()
    }

    func _testAuthProfileId() -> String {
        self.authProfileId
    }

    func _testSaveProviderAndModelConfig() async {
        await self.saveProviderAndModelConfig()
    }
}
#endif

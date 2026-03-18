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
            // For custom, pre-fill from customModelId if set
            self.selectedModel = self.customModelId
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
        self.writeApiKeyCredential()
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
                self.setupDetailMessage = "Gateway is starting up (may take a moment)..."
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
        self.writeApiKeyCredential()
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

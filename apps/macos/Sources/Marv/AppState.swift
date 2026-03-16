import AppKit
import Foundation
import Observation
import ServiceManagement
import SwiftUI

@MainActor
@Observable
final class AppState {
    private let isPreview: Bool
    private var isInitializing = true
    private var configWatcher: ConfigFileWatcher?

    private func ifNotPreview(_ action: () -> Void) {
        guard !self.isPreview else { return }
        action()
    }

    enum ConnectionMode: String {
        case unconfigured
        case local
        case remote
    }

    enum RemoteTransport: String {
        case ssh
        case direct
    }

    var isPaused: Bool {
        didSet { self.ifNotPreview { UserDefaults.standard.set(self.isPaused, forKey: pauseDefaultsKey) } }
    }

    var launchAtLogin: Bool {
        didSet {
            guard !self.isInitializing else { return }
            self.ifNotPreview { Task { AppStateStore.updateLaunchAtLogin(enabled: self.launchAtLogin) } }
        }
    }

    var onboardingSeen: Bool {
        didSet { self.ifNotPreview { UserDefaults.standard.set(self.onboardingSeen, forKey: onboardingSeenKey) }
        }
    }

    var debugPaneEnabled: Bool {
        didSet {
            self.ifNotPreview { UserDefaults.standard.set(self.debugPaneEnabled, forKey: debugPaneEnabledKey) }
        }
    }

    var iconAnimationsEnabled: Bool {
        didSet { self.ifNotPreview { UserDefaults.standard.set(
            self.iconAnimationsEnabled,
            forKey: iconAnimationsEnabledKey) } }
    }

    var showDockIcon: Bool {
        didSet {
            self.ifNotPreview {
                UserDefaults.standard.set(self.showDockIcon, forKey: showDockIconKey)
                AppActivationPolicy.apply(showDockIcon: self.showDockIcon)
            }
        }
    }

    /// Gateway-provided UI accent color (hex). Optional; clients provide a default.
    var seamColorHex: String?

    var iconOverride: IconOverrideSelection {
        didSet { self.ifNotPreview { UserDefaults.standard.set(self.iconOverride.rawValue, forKey: iconOverrideKey) } }
    }

    var isWorking: Bool = false
    var blinkTick: Int = 0
    var sendCelebrationTick: Int = 0
    var heartbeatsEnabled: Bool {
        didSet {
            self.ifNotPreview {
                UserDefaults.standard.set(self.heartbeatsEnabled, forKey: heartbeatsEnabledKey)
                Task { _ = await GatewayConnection.shared.setHeartbeatsEnabled(self.heartbeatsEnabled) }
            }
        }
    }

    var connectionMode: ConnectionMode {
        didSet {
            self.ifNotPreview { UserDefaults.standard.set(self.connectionMode.rawValue, forKey: connectionModeKey) }
            self.syncGatewayConfigIfNeeded()
        }
    }

    var remoteTransport: RemoteTransport {
        didSet { self.syncGatewayConfigIfNeeded() }
    }

    var remoteTarget: String {
        didSet {
            self.ifNotPreview { UserDefaults.standard.set(self.remoteTarget, forKey: remoteTargetKey) }
            self.syncGatewayConfigIfNeeded()
        }
    }

    var remoteUrl: String {
        didSet { self.syncGatewayConfigIfNeeded() }
    }

    var remoteIdentity: String {
        didSet { self.ifNotPreview { UserDefaults.standard.set(self.remoteIdentity, forKey: remoteIdentityKey) } }
    }

    var remoteProjectRoot: String {
        didSet { self.ifNotPreview { UserDefaults.standard.set(self.remoteProjectRoot, forKey: remoteProjectRootKey) } }
    }

    var remoteCliPath: String {
        didSet { self.ifNotPreview { UserDefaults.standard.set(self.remoteCliPath, forKey: remoteCliPathKey) } }
    }

    init(preview: Bool = false) {
        let isPreview = preview || ProcessInfo.processInfo.isRunningTests
        self.isPreview = isPreview
        if !isPreview {
            migrateLegacyDefaults()
        }
        let onboardingSeen = UserDefaults.standard.bool(forKey: onboardingSeenKey)
        self.isPaused = UserDefaults.standard.bool(forKey: pauseDefaultsKey)
        self.launchAtLogin = false
        self.onboardingSeen = onboardingSeen
        self.debugPaneEnabled = UserDefaults.standard.bool(forKey: debugPaneEnabledKey)
        if let storedIconAnimations = UserDefaults.standard.object(forKey: iconAnimationsEnabledKey) as? Bool {
            self.iconAnimationsEnabled = storedIconAnimations
        } else {
            self.iconAnimationsEnabled = true
            UserDefaults.standard.set(true, forKey: iconAnimationsEnabledKey)
        }
        self.showDockIcon = UserDefaults.standard.bool(forKey: showDockIconKey)
        self.seamColorHex = nil
        if let storedHeartbeats = UserDefaults.standard.object(forKey: heartbeatsEnabledKey) as? Bool {
            self.heartbeatsEnabled = storedHeartbeats
        } else {
            self.heartbeatsEnabled = true
            UserDefaults.standard.set(true, forKey: heartbeatsEnabledKey)
        }
        if let storedOverride = UserDefaults.standard.string(forKey: iconOverrideKey),
           let selection = IconOverrideSelection(rawValue: storedOverride)
        {
            self.iconOverride = selection
        } else {
            self.iconOverride = .system
            UserDefaults.standard.set(IconOverrideSelection.system.rawValue, forKey: iconOverrideKey)
        }

        let configRoot = MarvConfigFile.loadDict()
        let configRemoteUrl = GatewayRemoteConfig.resolveUrlString(root: configRoot)
        let configRemoteTransport = GatewayRemoteConfig.resolveTransport(root: configRoot)
        let resolvedConnectionMode = ConnectionModeResolver.resolve(root: configRoot).mode
        self.remoteTransport = configRemoteTransport
        self.connectionMode = resolvedConnectionMode

        let storedRemoteTarget = UserDefaults.standard.string(forKey: remoteTargetKey) ?? ""
        if resolvedConnectionMode == .remote,
           configRemoteTransport != .direct,
           storedRemoteTarget.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           let host = AppState.remoteHost(from: configRemoteUrl)
        {
            self.remoteTarget = "\(NSUserName())@\(host)"
        } else {
            self.remoteTarget = storedRemoteTarget
        }
        self.remoteUrl = configRemoteUrl ?? ""
        self.remoteIdentity = UserDefaults.standard.string(forKey: remoteIdentityKey) ?? ""
        self.remoteProjectRoot = UserDefaults.standard.string(forKey: remoteProjectRootKey) ?? ""
        self.remoteCliPath = UserDefaults.standard.string(forKey: remoteCliPathKey) ?? ""
        if !self.isPreview {
            Task.detached(priority: .utility) { [weak self] in
                let current = await LaunchAgentManager.status()
                await MainActor.run { [weak self] in self?.launchAtLogin = current }
            }
        }

        self.isInitializing = false
        if !self.isPreview {
            self.startConfigWatcher()
        }
    }

    @MainActor
    deinit {
        self.configWatcher?.stop()
    }

    private static func remoteHost(from urlString: String?) -> String? {
        guard let raw = urlString?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty,
              let url = URL(string: raw),
              let host = url.host?.trimmingCharacters(in: .whitespacesAndNewlines),
              !host.isEmpty
        else {
            return nil
        }
        return host
    }

    private static func sanitizeSSHTarget(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("ssh ") {
            return trimmed.replacingOccurrences(of: "ssh ", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return trimmed
    }

    private func startConfigWatcher() {
        let configUrl = MarvConfigFile.url()
        self.configWatcher = ConfigFileWatcher(url: configUrl) { [weak self] in
            Task { @MainActor in
                self?.applyConfigFromDisk()
            }
        }
        self.configWatcher?.start()
    }

    private func applyConfigFromDisk() {
        let root = MarvConfigFile.loadDict()
        self.applyConfigOverrides(root)
    }

    private func applyConfigOverrides(_ root: [String: Any]) {
        let gateway = root["gateway"] as? [String: Any]
        let modeRaw = (gateway?["mode"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
        let remoteUrl = GatewayRemoteConfig.resolveUrlString(root: root)
        let hasRemoteUrl = !(remoteUrl?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty ?? true)
        let remoteTransport = GatewayRemoteConfig.resolveTransport(root: root)

        let desiredMode: ConnectionMode? = switch modeRaw {
        case "local":
            .local
        case "remote":
            .remote
        case "unconfigured":
            .unconfigured
        default:
            nil
        }

        if let desiredMode {
            if desiredMode != self.connectionMode {
                self.connectionMode = desiredMode
            }
        } else if hasRemoteUrl, self.connectionMode != .remote {
            self.connectionMode = .remote
        }

        if remoteTransport != self.remoteTransport {
            self.remoteTransport = remoteTransport
        }
        let remoteUrlText = remoteUrl ?? ""
        if remoteUrlText != self.remoteUrl {
            self.remoteUrl = remoteUrlText
        }

        let targetMode = desiredMode ?? self.connectionMode
        if targetMode == .remote,
           remoteTransport != .direct,
           let host = AppState.remoteHost(from: remoteUrl)
        {
            self.updateRemoteTarget(host: host)
        }
    }

    private func updateRemoteTarget(host: String) {
        let trimmed = self.remoteTarget.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let parsed = CommandResolver.parseSSHTarget(trimmed) else { return }
        let trimmedUser = parsed.user?.trimmingCharacters(in: .whitespacesAndNewlines)
        let user = (trimmedUser?.isEmpty ?? true) ? nil : trimmedUser
        let port = parsed.port
        let assembled: String = if let user {
            port == 22 ? "\(user)@\(host)" : "\(user)@\(host):\(port)"
        } else {
            port == 22 ? host : "\(host):\(port)"
        }
        if assembled != self.remoteTarget {
            self.remoteTarget = assembled
        }
    }

    private func syncGatewayConfigIfNeeded() {
        guard !self.isPreview, !self.isInitializing else { return }

        let connectionMode = self.connectionMode
        let remoteTarget = self.remoteTarget
        let remoteIdentity = self.remoteIdentity
        let remoteTransport = self.remoteTransport
        let remoteUrl = self.remoteUrl
        let desiredMode: String? = switch connectionMode {
        case .local:
            "local"
        case .remote:
            "remote"
        case .unconfigured:
            nil
        }
        let remoteHost = connectionMode == .remote
            ? CommandResolver.parseSSHTarget(remoteTarget)?.host
            : nil

        Task { @MainActor in
            // Keep app-only connection settings local to avoid overwriting remote gateway config.
            var root = MarvConfigFile.loadDict()
            var gateway = root["gateway"] as? [String: Any] ?? [:]
            var changed = false

            let currentMode = (gateway["mode"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let desiredMode {
                if currentMode != desiredMode {
                    gateway["mode"] = desiredMode
                    changed = true
                }
            } else if currentMode != nil {
                gateway.removeValue(forKey: "mode")
                changed = true
            }

            if connectionMode == .remote {
                var remote = gateway["remote"] as? [String: Any] ?? [:]
                var remoteChanged = false

                if remoteTransport == .direct {
                    let trimmedUrl = remoteUrl.trimmingCharacters(in: .whitespacesAndNewlines)
                    if trimmedUrl.isEmpty {
                        if remote["url"] != nil {
                            remote.removeValue(forKey: "url")
                            remoteChanged = true
                        }
                    } else if let normalizedUrl = GatewayRemoteConfig.normalizeGatewayUrlString(trimmedUrl) {
                        if (remote["url"] as? String) != normalizedUrl {
                            remote["url"] = normalizedUrl
                            remoteChanged = true
                        }
                    }
                    if (remote["transport"] as? String) != RemoteTransport.direct.rawValue {
                        remote["transport"] = RemoteTransport.direct.rawValue
                        remoteChanged = true
                    }
                } else {
                    if remote["transport"] != nil {
                        remote.removeValue(forKey: "transport")
                        remoteChanged = true
                    }
                    if let host = remoteHost {
                        let existingUrl = (remote["url"] as? String)?
                            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                        let parsedExisting = existingUrl.isEmpty ? nil : URL(string: existingUrl)
                        let scheme = parsedExisting?.scheme?.isEmpty == false ? parsedExisting?.scheme : "ws"
                        let port = parsedExisting?.port ?? 18789
                        let desiredUrl = "\(scheme ?? "ws")://\(host):\(port)"
                        if existingUrl != desiredUrl {
                            remote["url"] = desiredUrl
                            remoteChanged = true
                        }
                    }

                    let sanitizedTarget = Self.sanitizeSSHTarget(remoteTarget)
                    if !sanitizedTarget.isEmpty {
                        if (remote["sshTarget"] as? String) != sanitizedTarget {
                            remote["sshTarget"] = sanitizedTarget
                            remoteChanged = true
                        }
                    } else if remote["sshTarget"] != nil {
                        remote.removeValue(forKey: "sshTarget")
                        remoteChanged = true
                    }

                    let trimmedIdentity = remoteIdentity.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmedIdentity.isEmpty {
                        if (remote["sshIdentity"] as? String) != trimmedIdentity {
                            remote["sshIdentity"] = trimmedIdentity
                            remoteChanged = true
                        }
                    } else if remote["sshIdentity"] != nil {
                        remote.removeValue(forKey: "sshIdentity")
                        remoteChanged = true
                    }
                }

                if remoteChanged {
                    gateway["remote"] = remote
                    changed = true
                }
            }

            guard changed else { return }
            if gateway.isEmpty {
                root.removeValue(forKey: "gateway")
            } else {
                root["gateway"] = gateway
            }
            MarvConfigFile.saveDict(root)
        }
    }

    func blinkOnce() {
        self.blinkTick &+= 1
    }

    func celebrateSend() {
        self.sendCelebrationTick &+= 1
    }

    func setWorking(_ working: Bool) {
        self.isWorking = working
    }
}

extension AppState {
    static var preview: AppState {
        let state = AppState(preview: true)
        state.isPaused = false
        state.launchAtLogin = true
        state.onboardingSeen = true
        state.debugPaneEnabled = true
        state.iconAnimationsEnabled = true
        state.showDockIcon = true
        state.iconOverride = .system
        state.heartbeatsEnabled = true
        state.connectionMode = .local
        state.remoteTransport = .ssh
        state.remoteTarget = "user@example.com"
        state.remoteUrl = "wss://gateway.example.ts.net"
        state.remoteIdentity = "~/.ssh/id_ed25519"
        state.remoteProjectRoot = "~/Projects/marv"
        state.remoteCliPath = ""
        return state
    }
}

@MainActor
enum AppStateStore {
    static let shared = AppState()
    static var isPausedFlag: Bool {
        UserDefaults.standard.bool(forKey: pauseDefaultsKey)
    }

    static func updateLaunchAtLogin(enabled: Bool) {
        Task.detached(priority: .utility) {
            await LaunchAgentManager.set(enabled: enabled, bundlePath: Bundle.main.bundlePath)
        }
    }
}

@MainActor
enum AppActivationPolicy {
    static func apply(showDockIcon: Bool) {
        _ = showDockIcon
        DockIconManager.shared.updateDockVisibility()
    }
}

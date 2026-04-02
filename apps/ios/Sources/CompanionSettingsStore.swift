import Foundation

struct CompanionGatewaySettings: Equatable, Sendable {
    var gatewayURL: String
    var sessionKey: String
    var cameraNodeEnabled: Bool

    static let `default` = CompanionGatewaySettings(
        gatewayURL: "",
        sessionKey: "main",
        cameraNodeEnabled: false)
}

enum CompanionSettingsStore {
    private static var defaults: UserDefaults { .standard }
    private static let gatewayURLKey = "companion.gateway.url"
    private static let sessionKeyKey = "companion.session.key"
    private static let cameraNodeEnabledKey = "companion.cameraNode.enabled"
    private static let gatewaySecretsService = "ai.marv.ios.gateway"
    private static let gatewayTokenAccount = "shared-token"
    private static let gatewayPasswordAccount = "shared-password"

    static func loadSettings() -> CompanionGatewaySettings {
        let gatewayURL = (self.defaults.string(forKey: self.gatewayURLKey) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let sessionKey = (self.defaults.string(forKey: self.sessionKeyKey) ?? "main")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return CompanionGatewaySettings(
            gatewayURL: gatewayURL,
            sessionKey: sessionKey.isEmpty ? "main" : sessionKey,
            cameraNodeEnabled: self.defaults.bool(forKey: self.cameraNodeEnabledKey))
    }

    static func saveSettings(_ settings: CompanionGatewaySettings) {
        let gatewayURL = settings.gatewayURL.trimmingCharacters(in: .whitespacesAndNewlines)
        let sessionKey = settings.sessionKey.trimmingCharacters(in: .whitespacesAndNewlines)
        self.defaults.set(gatewayURL, forKey: self.gatewayURLKey)
        self.defaults.set(sessionKey.isEmpty ? "main" : sessionKey, forKey: self.sessionKeyKey)
        self.defaults.set(settings.cameraNodeEnabled, forKey: self.cameraNodeEnabledKey)
    }

    static func loadGatewayToken() -> String {
        KeychainStore.loadString(
            service: self.gatewaySecretsService,
            account: self.gatewayTokenAccount)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    static func saveGatewayToken(_ token: String) {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            _ = KeychainStore.delete(service: self.gatewaySecretsService, account: self.gatewayTokenAccount)
        } else {
            _ = KeychainStore.saveString(
                trimmed,
                service: self.gatewaySecretsService,
                account: self.gatewayTokenAccount)
        }
    }

    static func loadGatewayPassword() -> String {
        KeychainStore.loadString(
            service: self.gatewaySecretsService,
            account: self.gatewayPasswordAccount)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    static func saveGatewayPassword(_ password: String) {
        let trimmed = password.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            _ = KeychainStore.delete(service: self.gatewaySecretsService, account: self.gatewayPasswordAccount)
        } else {
            _ = KeychainStore.saveString(
                trimmed,
                service: self.gatewaySecretsService,
                account: self.gatewayPasswordAccount)
        }
    }
}

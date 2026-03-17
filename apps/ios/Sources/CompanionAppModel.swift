import OpenClawKit
import OpenClawProtocol
import Foundation
import Observation
import SwiftUI

@MainActor
@Observable
final class CompanionAppModel {
    var gatewayURL = ""
    var gatewayToken = ""
    var gatewayPassword = ""
    var sessionKey = "main"
    var cameraNodeEnabled = false

    var operatorStatusText = "Not connected"
    var nodeStatusText = "Camera node disabled"
    var isConnecting = false
    var isForegroundActive = true

    let dashboard = DashboardState()
    let speechInput = SpeechInputModel()

    @ObservationIgnored
    let operatorGateway = GatewayNodeSession()
    @ObservationIgnored
    let nodeGateway = GatewayNodeSession()
    @ObservationIgnored
    private let cameraController = CameraController()
    @ObservationIgnored
    private let encoder = JSONEncoder()
    @ObservationIgnored
    private let decoder = JSONDecoder()
    @ObservationIgnored
    private var hasLoadedSettings = false

    var isOperatorConnected: Bool {
        self.operatorStatusText == "Connected"
    }

    func loadAndConnectIfPossible() async {
        if !self.hasLoadedSettings {
            self.loadPersistedSettings()
        }
        guard !self.gatewayURL.isEmpty else { return }
        await self.connect()
    }

    func loadPersistedSettings() {
        let settings = CompanionSettingsStore.loadSettings()
        self.gatewayURL = settings.gatewayURL
        self.sessionKey = settings.sessionKey
        self.cameraNodeEnabled = settings.cameraNodeEnabled
        self.gatewayToken = CompanionSettingsStore.loadGatewayToken()
        self.gatewayPassword = CompanionSettingsStore.loadGatewayPassword()
        self.hasLoadedSettings = true
        if !self.cameraNodeEnabled {
            self.nodeStatusText = "Camera node disabled"
        }
    }

    func saveSettings() {
        CompanionSettingsStore.saveSettings(
            CompanionGatewaySettings(
                gatewayURL: self.gatewayURL,
                sessionKey: self.sessionKey,
                cameraNodeEnabled: self.cameraNodeEnabled))
        CompanionSettingsStore.saveGatewayToken(self.gatewayToken)
        CompanionSettingsStore.saveGatewayPassword(self.gatewayPassword)
    }

    func saveAndReconnect() async {
        self.saveSettings()
        await self.connect()
    }

    func setScenePhase(_ phase: ScenePhase) {
        self.isForegroundActive = phase == .active
    }

    func connect() async {
        let trimmedURL = self.gatewayURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmedURL), let scheme = url.scheme, scheme == "ws" || scheme == "wss" else {
            self.operatorStatusText = "Enter a ws:// or wss:// gateway URL"
            return
        }
        self.isConnecting = true
        self.operatorStatusText = "Connecting"
        self.dashboard.errorText = nil

        await self.disconnectNodeOnly()

        do {
            try await self.operatorGateway.connect(
                url: url,
                token: self.nonEmpty(self.gatewayToken),
                password: self.nonEmpty(self.gatewayPassword),
                connectOptions: GatewayConnectOptions(
                    role: "operator",
                    scopes: ["operator.read", "operator.write"],
                    caps: [],
                    commands: [],
                    permissions: [:],
                    clientId: "marv-ios-companion",
                    clientMode: "companion-ui",
                    clientDisplayName: "Marv Companion",
                    includeDeviceIdentity: true),
                sessionBox: nil,
                onConnected: { [weak self] in
                    await MainActor.run {
                        self?.operatorStatusText = "Connected"
                    }
                },
                onDisconnected: { [weak self] reason in
                    await MainActor.run {
                        self?.operatorStatusText = "Disconnected: \(reason)"
                    }
                },
                onInvoke: { request in
                    BridgeInvokeResponse(
                        id: request.id,
                        ok: false,
                        error: OpenClawNodeError(code: .invalidRequest, message: "operator session cannot handle node invokes"))
                })
            self.operatorStatusText = "Connected"
            await self.refreshDashboard()
            if self.cameraNodeEnabled {
                await self.connectNode(url: url)
            }
        } catch {
            self.operatorStatusText = error.localizedDescription
            self.dashboard.errorText = error.localizedDescription
        }
        self.isConnecting = false
    }

    func disconnect() async {
        self.dashboard.resetForDisconnect()
        self.speechInput.stopListening()
        await self.operatorGateway.disconnect()
        await self.nodeGateway.disconnect()
        self.operatorStatusText = "Not connected"
        self.nodeStatusText = self.cameraNodeEnabled ? "Not connected" : "Camera node disabled"
    }

    func disconnectNodeOnly() async {
        await self.nodeGateway.disconnect()
        self.nodeStatusText = self.cameraNodeEnabled ? "Not connected" : "Camera node disabled"
    }

    func refreshDashboard() async {
        guard self.isOperatorConnected else {
            self.dashboard.resetForDisconnect()
            return
        }
        self.dashboard.isLoading = true
        self.dashboard.errorText = nil
        do {
            async let memory = self.requestDecoded("memory.stats", as: MemoryStatusSnapshot.self)
            async let knowledge = self.requestDecoded("knowledge.status", as: KnowledgeStatusSnapshot.self)
            async let proactive = self.requestDecoded("proactive.buffer", as: ProactiveStatusSnapshot.self)
            self.dashboard.memory = try await memory
            self.dashboard.knowledge = try await knowledge
            self.dashboard.proactive = try await proactive
        } catch {
            self.dashboard.errorText = error.localizedDescription
        }
        // Non-critical sections: fetch independently so one failure doesn't block others.
        async let sessionsResult: Void = self.fetchSessions()
        async let cronResult: Void = self.fetchCron()
        async let usageResult: Void = self.fetchUsage()
        _ = await (sessionsResult, cronResult, usageResult)
        self.dashboard.isLoading = false
    }

    private func fetchSessions() async {
        do {
            let params = #"{"includeGlobal":false,"includeUnknown":false,"limit":20}"#
            self.dashboard.sessions = try await self.requestDecoded("sessions.list", as: SessionsListResult.self, paramsJSON: params)
        } catch {
            // Non-critical; silently skip.
        }
    }

    private func fetchCron() async {
        do {
            self.dashboard.cronStatus = try await self.requestDecoded("cron.status", as: CronStatusSnapshot.self)
            let params = #"{"includeDisabled":true}"#
            self.dashboard.cronJobs = try await self.requestDecoded("cron.list", as: CronListResult.self, paramsJSON: params)
        } catch {
            // Non-critical.
        }
    }

    private func fetchUsage() async {
        do {
            // Fetch last 7 days of cost data.
            let calendar = Calendar.current
            let now = Date()
            let start = calendar.date(byAdding: .day, value: -7, to: now) ?? now
            let fmt = ISO8601DateFormatter()
            fmt.formatOptions = [.withFullDate]
            let params = #"{"startDate":"\#(fmt.string(from: start))","endDate":"\#(fmt.string(from: now))"}"#
            self.dashboard.usage = try await self.requestDecoded("usage.cost", as: CostUsageSummary.self, paramsJSON: params)
        } catch {
            // Non-critical.
        }
    }

    func sendTranscript() async {
        let text = self.speechInput.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        do {
            _ = try await CompanionChatTransport(gateway: self.operatorGateway).sendMessage(
                sessionKey: self.resolvedSessionKey,
                message: text,
                thinking: "low",
                idempotencyKey: UUID().uuidString,
                attachments: [])
            self.speechInput.reset()
        } catch {
            self.speechInput.errorText = error.localizedDescription
        }
    }

    private func connectNode(url: URL) async {
        self.nodeStatusText = "Connecting camera node"
        do {
            try await self.nodeGateway.connect(
                url: url,
                token: self.nonEmpty(self.gatewayToken),
                password: self.nonEmpty(self.gatewayPassword),
                connectOptions: GatewayConnectOptions(
                    role: "node",
                    scopes: [],
                    caps: ["camera"],
                    commands: ["camera.list", "camera.snap"],
                    permissions: ["camera": true],
                    clientId: "marv-ios-camera-node",
                    clientMode: "companion-node",
                    clientDisplayName: "Marv Camera Node",
                    includeDeviceIdentity: true),
                sessionBox: nil,
                onConnected: { [weak self] in
                    await MainActor.run {
                        self?.nodeStatusText = "Camera node connected"
                    }
                },
                onDisconnected: { [weak self] reason in
                    await MainActor.run {
                        self?.nodeStatusText = "Disconnected: \(reason)"
                    }
                },
                onInvoke: { [weak self] request in
                    guard let self else {
                        return BridgeInvokeResponse(
                            id: request.id,
                            ok: false,
                            error: OpenClawNodeError(code: .unavailable, message: "app model unavailable"))
                    }
                    return await self.handleNodeInvoke(request)
                })
            self.nodeStatusText = "Camera node connected"
        } catch {
            self.nodeStatusText = error.localizedDescription
        }
    }

    private func handleNodeInvoke(_ request: BridgeInvokeRequest) async -> BridgeInvokeResponse {
        guard self.cameraNodeEnabled else {
            return BridgeInvokeResponse(
                id: request.id,
                ok: false,
                error: OpenClawNodeError(code: .unavailable, message: "camera node disabled in settings"))
        }
        guard self.isForegroundActive else {
            return BridgeInvokeResponse(
                id: request.id,
                ok: false,
                error: OpenClawNodeError(code: .backgroundUnavailable, message: "bring the iPhone app to the foreground"))
        }

        do {
            switch request.command {
            case "camera.list":
                let payload = CameraListPayload(devices: await self.cameraController.listDevices())
                return try await self.successResponse(id: request.id, payload: payload)
            case "camera.snap":
                let params = try self.decodeParams(
                    request.paramsJSON,
                    fallback: OpenClawCameraSnapParams())
                let result = try await self.cameraController.snap(params: params)
                let payload = CameraSnapPayload(
                    format: result.format,
                    base64: result.base64,
                    width: result.width,
                    height: result.height)
                return try await self.successResponse(id: request.id, payload: payload)
            default:
                return BridgeInvokeResponse(
                    id: request.id,
                    ok: false,
                    error: OpenClawNodeError(code: .invalidRequest, message: "unsupported command \(request.command)"))
            }
        } catch {
            return BridgeInvokeResponse(
                id: request.id,
                ok: false,
                error: OpenClawNodeError(code: .unavailable, message: error.localizedDescription))
        }
    }

    private func requestDecoded<T: Decodable>(_ method: String, as _: T.Type, paramsJSON: String? = nil) async throws -> T {
        let data = try await self.operatorGateway.request(method: method, paramsJSON: paramsJSON, timeoutSeconds: 15)
        return try self.decoder.decode(T.self, from: data)
    }

    private func successResponse<T: Encodable>(id: String, payload: T) async throws -> BridgeInvokeResponse {
        let data = try self.encoder.encode(payload)
        let json = String(data: data, encoding: .utf8)
        return BridgeInvokeResponse(id: id, ok: true, payloadJSON: json, error: nil)
    }

    private func decodeParams<T: Decodable>(_ json: String?, fallback: T) throws -> T {
        guard let json, !json.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return fallback
        }
        guard let data = json.data(using: .utf8) else {
            throw NSError(domain: "MarvCompanion", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Failed to decode params as UTF-8",
            ])
        }
        return try self.decoder.decode(T.self, from: data)
    }

    private var resolvedSessionKey: String {
        let trimmed = self.sessionKey.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "main" : trimmed
    }

    private func nonEmpty(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

private struct CameraListPayload: Encodable {
    var devices: [CameraController.CameraDeviceInfo]
}

private struct CameraSnapPayload: Encodable {
    var format: String
    var base64: String
    var width: Int
    var height: Int
}

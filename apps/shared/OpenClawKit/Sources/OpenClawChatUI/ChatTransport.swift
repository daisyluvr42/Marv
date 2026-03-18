import Foundation

// MARK: - Exec Approval Models

/// A pending exec/escalation approval request from the gateway.
public struct OpenClawExecApprovalRequest: Sendable, Identifiable {
    public let id: String
    public let command: String
    public let kind: String?
    public let taskId: String?
    public let cwd: String?
    public let host: String?
    public let agentId: String?
    public let sessionKey: String?
    public let security: String?
    public let ask: String?
    public let resolvedPath: String?
    public let createdAtMs: Double?
    public let expiresAtMs: Double?

    public init(
        id: String,
        command: String,
        kind: String? = nil,
        taskId: String? = nil,
        cwd: String? = nil,
        host: String? = nil,
        agentId: String? = nil,
        sessionKey: String? = nil,
        security: String? = nil,
        ask: String? = nil,
        resolvedPath: String? = nil,
        createdAtMs: Double? = nil,
        expiresAtMs: Double? = nil)
    {
        self.id = id
        self.command = command
        self.kind = kind
        self.taskId = taskId
        self.cwd = cwd
        self.host = host
        self.agentId = agentId
        self.sessionKey = sessionKey
        self.security = security
        self.ask = ask
        self.resolvedPath = resolvedPath
        self.createdAtMs = createdAtMs
        self.expiresAtMs = expiresAtMs
    }

    /// Whether this is a permission-escalation request (vs. a plain exec approval).
    public var isEscalation: Bool {
        self.kind == "permission-escalation"
    }
}

// MARK: - Transport Events

public enum OpenClawChatTransportEvent: Sendable {
    case health(ok: Bool)
    case tick
    case chat(OpenClawChatEventPayload)
    case agent(OpenClawAgentEventPayload)
    case execApprovalRequested(OpenClawExecApprovalRequest)
    case execApprovalResolved(id: String, decision: String)
    case seqGap
}

public protocol OpenClawChatTransport: Sendable {
    func requestHistory(sessionKey: String) async throws -> OpenClawChatHistoryPayload
    func sendMessage(
        sessionKey: String,
        message: String,
        thinking: String,
        idempotencyKey: String,
        attachments: [OpenClawChatAttachmentPayload]) async throws -> OpenClawChatSendResponse

    func abortRun(sessionKey: String, runId: String) async throws
    func listSessions(limit: Int?) async throws -> OpenClawChatSessionsListResponse

    func requestHealth(timeoutMs: Int) async throws -> Bool
    func events() -> AsyncStream<OpenClawChatTransportEvent>

    func setActiveSessionKey(_ sessionKey: String) async throws

    /// Resolve a pending exec/escalation approval request.
    func resolveExecApproval(id: String, decision: String) async throws
}

extension OpenClawChatTransport {
    public func setActiveSessionKey(_: String) async throws {}

    public func abortRun(sessionKey _: String, runId _: String) async throws {
        throw NSError(
            domain: "OpenClawChatTransport",
            code: 0,
            userInfo: [NSLocalizedDescriptionKey: "chat.abort not supported by this transport"])
    }

    public func listSessions(limit _: Int?) async throws -> OpenClawChatSessionsListResponse {
        throw NSError(
            domain: "OpenClawChatTransport",
            code: 0,
            userInfo: [NSLocalizedDescriptionKey: "sessions.list not supported by this transport"])
    }

    public func resolveExecApproval(id _: String, decision _: String) async throws {
        throw NSError(
            domain: "OpenClawChatTransport",
            code: 0,
            userInfo: [NSLocalizedDescriptionKey: "exec.approval.resolve not supported by this transport"])
    }
}

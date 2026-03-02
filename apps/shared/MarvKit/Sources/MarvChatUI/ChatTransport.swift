import Foundation

public enum MarvChatTransportEvent: Sendable {
    case health(ok: Bool)
    case tick
    case chat(MarvChatEventPayload)
    case agent(MarvAgentEventPayload)
    case seqGap
}

public protocol MarvChatTransport: Sendable {
    func requestHistory(sessionKey: String) async throws -> MarvChatHistoryPayload
    func sendMessage(
        sessionKey: String,
        message: String,
        thinking: String,
        idempotencyKey: String,
        attachments: [MarvChatAttachmentPayload]) async throws -> MarvChatSendResponse

    func abortRun(sessionKey: String, runId: String) async throws
    func listSessions(limit: Int?) async throws -> MarvChatSessionsListResponse

    func requestHealth(timeoutMs: Int) async throws -> Bool
    func events() -> AsyncStream<MarvChatTransportEvent>

    func setActiveSessionKey(_ sessionKey: String) async throws
}

extension MarvChatTransport {
    public func setActiveSessionKey(_: String) async throws {}

    public func abortRun(sessionKey _: String, runId _: String) async throws {
        throw NSError(
            domain: "MarvChatTransport",
            code: 0,
            userInfo: [NSLocalizedDescriptionKey: "chat.abort not supported by this transport"])
    }

    public func listSessions(limit _: Int?) async throws -> MarvChatSessionsListResponse {
        throw NSError(
            domain: "MarvChatTransport",
            code: 0,
            userInfo: [NSLocalizedDescriptionKey: "sessions.list not supported by this transport"])
    }
}

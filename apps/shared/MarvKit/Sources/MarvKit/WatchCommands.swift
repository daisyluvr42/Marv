import Foundation

public enum MarvWatchCommand: String, Codable, Sendable {
    case status = "watch.status"
    case notify = "watch.notify"
}

public struct MarvWatchStatusPayload: Codable, Sendable, Equatable {
    public var supported: Bool
    public var paired: Bool
    public var appInstalled: Bool
    public var reachable: Bool
    public var activationState: String

    public init(
        supported: Bool,
        paired: Bool,
        appInstalled: Bool,
        reachable: Bool,
        activationState: String)
    {
        self.supported = supported
        self.paired = paired
        self.appInstalled = appInstalled
        self.reachable = reachable
        self.activationState = activationState
    }
}

public struct MarvWatchNotifyParams: Codable, Sendable, Equatable {
    public var title: String
    public var body: String
    public var priority: MarvNotificationPriority?

    public init(title: String, body: String, priority: MarvNotificationPriority? = nil) {
        self.title = title
        self.body = body
        self.priority = priority
    }
}

public struct MarvWatchNotifyPayload: Codable, Sendable, Equatable {
    public var deliveredImmediately: Bool
    public var queuedForDelivery: Bool
    public var transport: String

    public init(deliveredImmediately: Bool, queuedForDelivery: Bool, transport: String) {
        self.deliveredImmediately = deliveredImmediately
        self.queuedForDelivery = queuedForDelivery
        self.transport = transport
    }
}

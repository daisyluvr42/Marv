import Foundation

public enum MarvRemindersCommand: String, Codable, Sendable {
    case list = "reminders.list"
    case add = "reminders.add"
}

public enum MarvReminderStatusFilter: String, Codable, Sendable {
    case incomplete
    case completed
    case all
}

public struct MarvRemindersListParams: Codable, Sendable, Equatable {
    public var status: MarvReminderStatusFilter?
    public var limit: Int?

    public init(status: MarvReminderStatusFilter? = nil, limit: Int? = nil) {
        self.status = status
        self.limit = limit
    }
}

public struct MarvRemindersAddParams: Codable, Sendable, Equatable {
    public var title: String
    public var dueISO: String?
    public var notes: String?
    public var listId: String?
    public var listName: String?

    public init(
        title: String,
        dueISO: String? = nil,
        notes: String? = nil,
        listId: String? = nil,
        listName: String? = nil)
    {
        self.title = title
        self.dueISO = dueISO
        self.notes = notes
        self.listId = listId
        self.listName = listName
    }
}

public struct MarvReminderPayload: Codable, Sendable, Equatable {
    public var identifier: String
    public var title: String
    public var dueISO: String?
    public var completed: Bool
    public var listName: String?

    public init(
        identifier: String,
        title: String,
        dueISO: String? = nil,
        completed: Bool,
        listName: String? = nil)
    {
        self.identifier = identifier
        self.title = title
        self.dueISO = dueISO
        self.completed = completed
        self.listName = listName
    }
}

public struct MarvRemindersListPayload: Codable, Sendable, Equatable {
    public var reminders: [MarvReminderPayload]

    public init(reminders: [MarvReminderPayload]) {
        self.reminders = reminders
    }
}

public struct MarvRemindersAddPayload: Codable, Sendable, Equatable {
    public var reminder: MarvReminderPayload

    public init(reminder: MarvReminderPayload) {
        self.reminder = reminder
    }
}

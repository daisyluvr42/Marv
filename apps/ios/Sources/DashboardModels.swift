import Foundation
import Observation

struct MemoryStatusSnapshot: Decodable, Sendable {
    var agentId: String
    var backend: String
    var citations: String
    var autoRecallEnabled: Bool
    var knowledgeEnabled: Bool
    var runtimeIngestEnabled: Bool
    var totalItems: Int
    var tiers: [String: Int]
    var recordKinds: [String: Int]
    var archiveEvents: Int
}

struct KnowledgeVaultStatus: Decodable, Sendable {
    var name: String
    var path: String
    var registryId: String
    var exclude: [String]
    var fileCount: Int
    var chunkCount: Int
    var lastScanAt: Double?
}

struct KnowledgeStatusSnapshot: Decodable, Sendable {
    var agentId: String
    var enabled: Bool
    var autoSyncOnSearch: Bool
    var autoSyncOnBoot: Bool
    var syncIntervalMs: Int?
    var vaultCount: Int
    var totalFiles: Int
    var totalChunks: Int
    var lastScanAt: Double?
    var vaults: [KnowledgeVaultStatus]
}

struct ProactiveDeliverySnapshot: Decodable, Sendable {
    var channel: String
    var to: String?
}

struct ProactiveStatusSnapshot: Decodable, Sendable {
    var agentId: String
    var enabled: Bool
    var checkEveryMinutes: Int?
    var digestTimes: [String]
    var delivery: ProactiveDeliverySnapshot
    var totalEntries: Int
    var pendingEntries: Int
    var deliveredEntries: Int
    var urgentEntries: Int
    var lastFlushAt: Double?
}

// MARK: - Sessions

struct SessionEntry: Decodable, Sendable {
    var key: String
    var updatedAt: Double?
    var totalTokens: Int?
    var label: String?
    var displayName: String?
    var derivedTitle: String?

    var title: String {
        self.displayName ?? self.label ?? self.derivedTitle ?? self.key
    }
}

struct SessionsListResult: Decodable, Sendable {
    var sessions: [SessionEntry]
}

// MARK: - Cron

struct CronStatusSnapshot: Decodable, Sendable {
    var enabled: Bool
    var jobCount: Int
    var nextWakeAtMs: Double?
}

struct CronJob: Decodable, Sendable {
    var id: String
    var label: String?
    var schedule: String?
    var enabled: Bool
    var lastRunAt: Double?
    var nextRunAt: Double?
    var lastError: String?
}

struct CronListResult: Decodable, Sendable {
    var jobs: [CronJob]
}

// MARK: - Usage / Cost

struct CostDayEntry: Decodable, Sendable {
    var date: String
    var totalCost: Double
    var totalTokens: Int
}

struct CostUsageTotals: Decodable, Sendable {
    var input: Int
    var output: Int
    var cacheRead: Int
    var cacheWrite: Int
    var totalTokens: Int
    var totalCost: Double
}

struct CostUsageSummary: Decodable, Sendable {
    var updatedAt: Double
    var days: Int
    var daily: [CostDayEntry]
    var totals: CostUsageTotals

    var totalCost: Double {
        self.totals.totalCost
    }
}

// MARK: - Dashboard state

@MainActor
@Observable
final class DashboardState {
    var isLoading = false
    var errorText: String?
    var memory: MemoryStatusSnapshot?
    var knowledge: KnowledgeStatusSnapshot?
    var proactive: ProactiveStatusSnapshot?
    var sessions: SessionsListResult?
    var cronStatus: CronStatusSnapshot?
    var cronJobs: CronListResult?
    var usage: CostUsageSummary?

    func resetForDisconnect() {
        self.isLoading = false
        self.errorText = nil
        self.memory = nil
        self.knowledge = nil
        self.proactive = nil
        self.sessions = nil
        self.cronStatus = nil
        self.cronJobs = nil
        self.usage = nil
    }
}

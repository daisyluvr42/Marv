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

@MainActor
@Observable
final class DashboardState {
    var isLoading = false
    var errorText: String?
    var memory: MemoryStatusSnapshot?
    var knowledge: KnowledgeStatusSnapshot?
    var proactive: ProactiveStatusSnapshot?

    func resetForDisconnect() {
        self.isLoading = false
        self.errorText = nil
        self.memory = nil
        self.knowledge = nil
        self.proactive = nil
    }
}

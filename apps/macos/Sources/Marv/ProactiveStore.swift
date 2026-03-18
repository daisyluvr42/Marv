import Foundation
import Observation

/// Tracks proactive digest buffer state from the gateway.
@MainActor
@Observable
final class ProactiveStore {
    static let shared = ProactiveStore()

    struct Snapshot: Codable, Sendable {
        let agentId: String
        let enabled: Bool
        let checkEveryMinutes: Int?
        let digestTimes: [String]
        let delivery: Delivery
        let totalEntries: Int
        let pendingEntries: Int
        let deliveredEntries: Int
        let urgentEntries: Int
        let lastFlushAt: Double?

        struct Delivery: Codable, Sendable {
            let channel: String
            let to: String?
        }
    }

    private(set) var snapshot: Snapshot?
    private(set) var isLoading = false
    private(set) var error: String?
    private var refreshTask: Task<Void, Never>?

    private init() {}

    func refresh() {
        self.refreshTask?.cancel()
        self.refreshTask = Task {
            self.isLoading = true
            self.error = nil
            do {
                let result: Snapshot = try await GatewayConnection.shared.requestDecoded(
                    method: .proactiveBuffer,
                    params: [:])
                self.snapshot = result
            } catch {
                self.error = error.localizedDescription
            }
            self.isLoading = false
        }
    }

    func flush() {
        Task {
            do {
                _ = try await GatewayConnection.shared.requestRaw(
                    method: .proactiveFlush,
                    params: [:],
                    timeoutMs: nil)
                // Refresh after flush to get updated counts
                self.refresh()
            } catch {
                self.error = error.localizedDescription
            }
        }
    }

    var hasUrgent: Bool {
        (self.snapshot?.urgentEntries ?? 0) > 0
    }

    var hasPending: Bool {
        (self.snapshot?.pendingEntries ?? 0) > 0
    }

    var nextDigestLabel: String? {
        guard let times = self.snapshot?.digestTimes, !times.isEmpty else { return nil }
        return times.first
    }
}

import SwiftUI

struct OperationsTab: View {
    @Environment(CompanionAppModel.self) private var appModel

    var body: some View {
        NavigationStack {
            List {
                if let errorText = self.appModel.dashboard.errorText, !errorText.isEmpty {
                    Section {
                        Text(errorText)
                            .foregroundStyle(.red)
                    }
                }

                sessionsSection
                cronSection
                usageSection

                if !self.appModel.isOperatorConnected && !self.appModel.dashboard.isLoading {
                    Section {
                        Text("Connect to the gateway to load operations data.")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .overlay {
                if self.appModel.dashboard.isLoading {
                    ProgressView()
                }
            }
            .navigationTitle("Operations")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Refresh") {
                        Task {
                            await self.appModel.refreshDashboard()
                        }
                    }
                    .disabled(!self.appModel.isOperatorConnected || self.appModel.dashboard.isLoading)
                }
            }
        }
    }

    // MARK: - Sessions

    @ViewBuilder
    private var sessionsSection: some View {
        if let sessions = self.appModel.dashboard.sessions {
            Section("Sessions (\(sessions.sessions.count))") {
                if sessions.sessions.isEmpty {
                    Text("No active sessions")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(sessions.sessions.prefix(10), id: \.key) { session in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(session.label ?? session.key)
                                    .font(.headline)
                                Spacer()
                                if let tokens = session.totalTokens, tokens > 0 {
                                    Text(Self.formatTokens(tokens))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            if session.label != nil {
                                Text(session.key)
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                            }
                            if let lastActive = session.lastActiveAt {
                                Text("Active \(Self.formatRelativeTime(lastActive))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
            }
        }
    }

    // MARK: - Cron

    @ViewBuilder
    private var cronSection: some View {
        if let cronStatus = self.appModel.dashboard.cronStatus {
            Section("Cron") {
                OpsKeyValueRow(label: "Enabled", value: cronStatus.enabled ? "On" : "Off")
                OpsKeyValueRow(label: "Jobs", value: "\(cronStatus.jobCount)")
                if let nextWake = cronStatus.nextWakeAtMs {
                    OpsKeyValueRow(label: "Next Wake", value: Self.formatTimestamp(nextWake))
                }
            }

            if let cronJobs = self.appModel.dashboard.cronJobs, !cronJobs.jobs.isEmpty {
                Section("Scheduled Jobs") {
                    ForEach(cronJobs.jobs, id: \.id) { job in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(job.label ?? job.id)
                                    .font(.headline)
                                Spacer()
                                Text(job.enabled ? "Active" : "Paused")
                                    .font(.caption)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 2)
                                    .background(job.enabled ? Color.green.opacity(0.15) : Color.gray.opacity(0.15))
                                    .clipShape(Capsule())
                                    .foregroundStyle(job.enabled ? .green : .secondary)
                            }
                            if let schedule = job.schedule {
                                Text(schedule)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            if let nextRun = job.nextRunAt {
                                Text("Next: \(Self.formatTimestamp(nextRun))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            if let lastError = job.lastError, !lastError.isEmpty {
                                Text("Error: \(lastError)")
                                    .font(.caption)
                                    .foregroundStyle(.red)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
            }
        }
    }

    // MARK: - Usage

    @ViewBuilder
    private var usageSection: some View {
        if let usage = self.appModel.dashboard.usage {
            Section("Usage (7 days)") {
                OpsKeyValueRow(label: "Total Cost", value: Self.formatCost(usage.totalCost))
                if let days = usage.days, !days.isEmpty {
                    ForEach(days.suffix(7).reversed(), id: \.date) { day in
                        HStack {
                            Text(day.date)
                                .font(.caption)
                            Spacer()
                            Text(Self.formatCost(day.totalCost))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Formatting

    private static func formatTimestamp(_ ms: Double) -> String {
        let date = Date(timeIntervalSince1970: ms / 1000.0)
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    private static func formatRelativeTime(_ ms: Double) -> String {
        let date = Date(timeIntervalSince1970: ms / 1000.0)
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }

    private static func formatTokens(_ tokens: Int) -> String {
        if tokens >= 1_000_000 {
            return String(format: "%.1fM tokens", Double(tokens) / 1_000_000)
        } else if tokens >= 1_000 {
            return String(format: "%.1fK tokens", Double(tokens) / 1_000)
        }
        return "\(tokens) tokens"
    }

    private static func formatCost(_ cost: Double) -> String {
        if cost < 0.01 && cost > 0 {
            return String(format: "$%.4f", cost)
        }
        return String(format: "$%.2f", cost)
    }
}

private struct OpsKeyValueRow: View {
    var label: String
    var value: String

    var body: some View {
        HStack {
            Text(self.label)
            Spacer()
            Text(self.value)
                .foregroundStyle(.secondary)
        }
    }
}

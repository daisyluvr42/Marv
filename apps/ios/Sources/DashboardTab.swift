import SwiftUI

struct DashboardTab: View {
    @Environment(CompanionAppModel.self) private var appModel

    private let columns = [
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12),
    ]

    var body: some View {
        NavigationStack {
            ZStack {
                Color(.systemGroupedBackground)
                    .ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        if let errorText = self.appModel.dashboard.errorText, !errorText.isEmpty {
                            DashboardMessageCard(text: errorText, tint: .red)
                        }

                        if !self.summaryCards.isEmpty {
                            LazyVGrid(columns: self.columns, spacing: 12) {
                                ForEach(self.summaryCards) { card in
                                    DashboardSummaryCard(card: card)
                                }
                            }
                        }

                        if let memory = self.appModel.dashboard.memory {
                            DashboardSectionCard(title: "Memory", subtitle: memory.backend) {
                                KeyValueRow(label: "Total Items", value: "\(memory.totalItems)")
                                KeyValueRow(label: "Archive Events", value: "\(memory.archiveEvents)")
                                KeyValueRow(label: "Auto Recall", value: memory.autoRecallEnabled ? "On" : "Off")
                                KeyValueRow(label: "Knowledge Sync", value: memory.knowledgeEnabled ? "On" : "Off")
                                KeyValueRow(label: "Runtime Ingest", value: memory.runtimeIngestEnabled ? "On" : "Off")
                            }
                        }

                        if let knowledge = self.appModel.dashboard.knowledge {
                            DashboardSectionCard(
                                title: "Knowledge",
                                subtitle: "\(knowledge.vaultCount) vaults, \(knowledge.totalFiles) files"
                            ) {
                                KeyValueRow(label: "Enabled", value: knowledge.enabled ? "On" : "Off")
                                KeyValueRow(label: "Files", value: "\(knowledge.totalFiles)")
                                KeyValueRow(label: "Chunks", value: "\(knowledge.totalChunks)")
                                if let lastScanAt = knowledge.lastScanAt {
                                    KeyValueRow(label: "Last Scan", value: Self.formatTimestamp(lastScanAt))
                                }
                                if !knowledge.vaults.isEmpty {
                                    Divider()
                                    VStack(alignment: .leading, spacing: 12) {
                                        ForEach(knowledge.vaults, id: \.registryId) { vault in
                                            VStack(alignment: .leading, spacing: 4) {
                                                Text(vault.name)
                                                    .font(.headline)
                                                Text(vault.path)
                                                    .font(.caption)
                                                    .foregroundStyle(.secondary)
                                                    .textSelection(.enabled)
                                                Text("\(vault.fileCount) files, \(vault.chunkCount) chunks")
                                                    .font(.caption)
                                                    .foregroundStyle(.secondary)
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if let proactive = self.appModel.dashboard.proactive {
                            DashboardSectionCard(
                                title: "Proactive",
                                subtitle: proactive.delivery.channel.capitalized
                            ) {
                                KeyValueRow(label: "Enabled", value: proactive.enabled ? "On" : "Off")
                                KeyValueRow(label: "Pending", value: "\(proactive.pendingEntries)")
                                KeyValueRow(label: "Delivered", value: "\(proactive.deliveredEntries)")
                                KeyValueRow(label: "Urgent", value: "\(proactive.urgentEntries)")
                                if let lastFlushAt = proactive.lastFlushAt {
                                    KeyValueRow(label: "Last Flush", value: Self.formatTimestamp(lastFlushAt))
                                }
                            }
                        }

                        if self.isEmptyStateVisible {
                            DashboardMessageCard(
                                text: "Connect to the gateway to load dashboard state.",
                                tint: .secondary
                            )
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 16)
                    .padding(.bottom, 24)
                }

                if self.appModel.dashboard.isLoading {
                    ProgressView()
                        .controlSize(.large)
                        .padding(20)
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18))
                }
            }
            .navigationTitle("Dashboard")
            .navigationBarTitleDisplayMode(.inline)
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

    private var isEmptyStateVisible: Bool {
        self.appModel.dashboard.memory == nil &&
            self.appModel.dashboard.knowledge == nil &&
            self.appModel.dashboard.proactive == nil &&
            !self.appModel.dashboard.isLoading
    }

    private var summaryCards: [DashboardSummaryCardData] {
        var cards: [DashboardSummaryCardData] = []

        if let memory = self.appModel.dashboard.memory {
            cards.append(.init(title: "Memory", value: "\(memory.totalItems)", detail: memory.backend))
        }
        if let knowledge = self.appModel.dashboard.knowledge {
            cards.append(.init(title: "Knowledge", value: "\(knowledge.totalFiles)", detail: "\(knowledge.vaultCount) vaults"))
        }
        if let proactive = self.appModel.dashboard.proactive {
            cards.append(.init(title: "Pending", value: "\(proactive.pendingEntries)", detail: "Proactive queue"))
        }
        if let sessions = self.appModel.dashboard.sessions {
            cards.append(.init(title: "Sessions", value: "\(sessions.sessions.count)", detail: "Recent activity"))
        }
        if let cronStatus = self.appModel.dashboard.cronStatus {
            cards.append(.init(title: "Cron", value: "\(cronStatus.jobCount)", detail: cronStatus.enabled ? "Enabled" : "Paused"))
        }
        if let usage = self.appModel.dashboard.usage {
            cards.append(.init(title: "Cost", value: Self.formatCost(usage.totalCost), detail: "Last \(usage.days) days"))
        }

        return cards
    }

    private static func formatTimestamp(_ value: Double) -> String {
        let date = Date(timeIntervalSince1970: value / 1000.0)
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    private static func formatCost(_ cost: Double) -> String {
        if cost < 0.01 && cost > 0 {
            return String(format: "$%.4f", cost)
        }
        return String(format: "$%.2f", cost)
    }
}

private struct DashboardSummaryCardData: Identifiable {
    let id = UUID()
    var title: String
    var value: String
    var detail: String
}

private struct DashboardSummaryCard: View {
    var card: DashboardSummaryCardData

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(self.card.title)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
            Text(self.card.value)
                .font(.title2.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            Text(self.card.detail)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 20))
    }
}

private struct DashboardSectionCard<Content: View>: View {
    var title: String
    var subtitle: String?
    private let content: Content

    init(title: String, subtitle: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.subtitle = subtitle
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text(self.title)
                    .font(.headline)
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            self.content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 22))
    }
}

private struct DashboardMessageCard: View {
    var text: String
    var tint: Color

    var body: some View {
        Text(self.text)
            .font(.subheadline)
            .foregroundStyle(self.tint)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 20))
    }
}

struct KeyValueRow: View {
    var label: String
    var value: String

    var body: some View {
        HStack {
            Text(self.label)
                .foregroundStyle(.primary)
            Spacer()
            Text(self.value)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }
}

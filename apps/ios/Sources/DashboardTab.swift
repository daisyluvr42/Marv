import SwiftUI

struct DashboardTab: View {
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

                if let memory = self.appModel.dashboard.memory {
                    Section("Memory") {
                        KeyValueRow(label: "Backend", value: memory.backend)
                        KeyValueRow(label: "Total Items", value: "\(memory.totalItems)")
                        KeyValueRow(label: "Archive Events", value: "\(memory.archiveEvents)")
                        KeyValueRow(label: "Auto Recall", value: memory.autoRecallEnabled ? "On" : "Off")
                        KeyValueRow(label: "Knowledge Sync", value: memory.knowledgeEnabled ? "On" : "Off")
                        KeyValueRow(label: "Runtime Ingest", value: memory.runtimeIngestEnabled ? "On" : "Off")
                    }
                }

                if let knowledge = self.appModel.dashboard.knowledge {
                    Section("Knowledge") {
                        KeyValueRow(label: "Enabled", value: knowledge.enabled ? "On" : "Off")
                        KeyValueRow(label: "Vaults", value: "\(knowledge.vaultCount)")
                        KeyValueRow(label: "Files", value: "\(knowledge.totalFiles)")
                        KeyValueRow(label: "Chunks", value: "\(knowledge.totalChunks)")
                        if let lastScanAt = knowledge.lastScanAt {
                            KeyValueRow(label: "Last Scan", value: Self.formatTimestamp(lastScanAt))
                        }
                        ForEach(knowledge.vaults, id: \.registryId) { vault in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(vault.name)
                                    .font(.headline)
                                Text(vault.path)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                                Text("\(vault.fileCount) files, \(vault.chunkCount) chunks")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }

                if let proactive = self.appModel.dashboard.proactive {
                    Section("Proactive") {
                        KeyValueRow(label: "Enabled", value: proactive.enabled ? "On" : "Off")
                        KeyValueRow(label: "Pending", value: "\(proactive.pendingEntries)")
                        KeyValueRow(label: "Delivered", value: "\(proactive.deliveredEntries)")
                        KeyValueRow(label: "Urgent", value: "\(proactive.urgentEntries)")
                        KeyValueRow(label: "Delivery", value: proactive.delivery.channel)
                        if let lastFlushAt = proactive.lastFlushAt {
                            KeyValueRow(label: "Last Flush", value: Self.formatTimestamp(lastFlushAt))
                        }
                    }
                }

                if self.appModel.dashboard.memory == nil &&
                    self.appModel.dashboard.knowledge == nil &&
                    self.appModel.dashboard.proactive == nil &&
                    !self.appModel.dashboard.isLoading
                {
                    Section {
                        Text("Connect to the gateway to load dashboard state.")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .overlay {
                if self.appModel.dashboard.isLoading {
                    ProgressView()
                }
            }
            .navigationTitle("Dashboard")
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

    private static func formatTimestamp(_ value: Double) -> String {
        let date = Date(timeIntervalSince1970: value / 1000.0)
        return date.formatted(date: .abbreviated, time: .shortened)
    }
}

private struct KeyValueRow: View {
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

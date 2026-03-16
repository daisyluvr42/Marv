import SwiftUI

/// Compact proactive digest queue view for the menubar popover.
@MainActor
struct ProactiveMenuView: View {
    @Bindable private var store = ProactiveStore.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Proactive", systemImage: "bell.badge")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                Spacer()

                if self.store.isLoading {
                    ProgressView()
                        .controlSize(.mini)
                } else {
                    Button {
                        self.store.refresh()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                            .font(.caption2)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                }
            }

            if let snapshot = self.store.snapshot {
                self.statsRow(snapshot)

                if let nextDigest = self.store.nextDigestLabel {
                    HStack(spacing: 4) {
                        Image(systemName: "clock")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Text("Next digest: \(nextDigest)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                if let lastFlush = snapshot.lastFlushAt, lastFlush > 0 {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.circle")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Text("Last flush: \(Self.relativeTime(lastFlush))")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                if snapshot.pendingEntries > 0 {
                    Button {
                        self.store.flush()
                    } label: {
                        Label("Flush Now", systemImage: "paperplane")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            } else if let error = self.store.error {
                Text(error)
                    .font(.caption2)
                    .foregroundStyle(.red)
            } else if !self.store.isLoading {
                Text("Not connected")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .onAppear { self.store.refresh() }
    }

    private func statsRow(_ snapshot: ProactiveStore.Snapshot) -> some View {
        HStack(spacing: 12) {
            self.statPill(
                count: snapshot.pendingEntries,
                label: "pending",
                color: snapshot.pendingEntries > 0 ? .orange : .secondary)
            self.statPill(
                count: snapshot.urgentEntries,
                label: "urgent",
                color: snapshot.urgentEntries > 0 ? .red : .secondary)
            self.statPill(
                count: snapshot.deliveredEntries,
                label: "delivered",
                color: .green)
        }
    }

    private func statPill(count: Int, label: String, color: Color) -> some View {
        VStack(spacing: 2) {
            Text("\(count)")
                .font(.system(.caption, design: .rounded).weight(.bold))
                .foregroundStyle(color)
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
    }

    private static func relativeTime(_ timestamp: Double) -> String {
        let date = Date(timeIntervalSince1970: timestamp / 1000)
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}

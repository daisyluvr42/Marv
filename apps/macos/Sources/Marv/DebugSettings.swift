import AppKit
import Observation
import SwiftUI
import UniformTypeIdentifiers

struct DebugSettings: View {
    @Bindable var state: AppState
    private let isPreview = ProcessInfo.processInfo.isPreview
    private let labelColumnWidth: CGFloat = 140
    @AppStorage(modelCatalogPathKey) private var modelCatalogPath: String = ModelCatalogLoader.defaultPath
    @AppStorage(modelCatalogReloadKey) private var modelCatalogReloadBump: Int = 0
    @AppStorage(iconOverrideKey) private var iconOverrideRaw: String = IconOverrideSelection.system.rawValue
    @State private var modelsCount: Int?
    @State private var modelsLoading = false
    @State private var modelsError: String?
    private let gatewayManager = GatewayProcessManager.shared
    private let healthStore = HealthStore.shared
    @State private var launchAgentWriteDisabled = GatewayLaunchAgentManager.isLaunchAgentWriteDisabled()
    @State private var launchAgentWriteError: String?
    @State private var gatewayRootInput: String = GatewayProcessManager.shared.projectRootPath()
    @State private var sessionStorePath: String = SessionLoader.defaultStorePath
    @State private var sessionStoreSaveError: String?
    @State private var portCheckInFlight = false
    @State private var portReports: [DebugActions.PortReport] = []
    @State private var portKillStatus: String?
    @State private var tunnelResetInFlight = false
    @State private var tunnelResetStatus: String?
    @State private var pendingKill: DebugActions.PortListener?
    @AppStorage(debugFileLogEnabledKey) private var diagnosticsFileLogEnabled: Bool = false
    @AppStorage(appLogLevelKey) private var appLogLevelRaw: String = AppLogLevel.default.rawValue

    init(state: AppState = AppStateStore.shared) {
        self.state = state
    }

    var body: some View {
        ScrollView(.vertical) {
            VStack(alignment: .leading, spacing: 14) {
                self.header

                self.launchdSection
                self.appInfoSection
                self.gatewaySection
                self.logsSection
                self.portsSection
                self.pathsSection
                self.quickActionsSection
                self.experimentsSection

                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 24)
            .padding(.vertical, 18)
            .groupBoxStyle(PlainSettingsGroupBoxStyle())
        }
        .task {
            guard !self.isPreview else { return }
            await self.reloadModels()
            self.loadSessionStorePath()
        }
        .alert(item: self.$pendingKill) { listener in
            Alert(
                title: Text("Kill \(listener.command) (\(listener.pid))?"),
                message: Text("This process looks expected for the current mode. Kill anyway?"),
                primaryButton: .destructive(Text("Kill")) {
                    Task { await self.killConfirmed(listener.pid) }
                },
                secondaryButton: .cancel())
        }
    }

    private var launchdSection: some View {
        GroupBox("Gateway startup") {
            VStack(alignment: .leading, spacing: 8) {
                Toggle("Attach only (skip launchd install)", isOn: self.$launchAgentWriteDisabled)
                    .onChange(of: self.launchAgentWriteDisabled) { _, newValue in
                        self.launchAgentWriteError = GatewayLaunchAgentManager.setLaunchAgentWriteDisabled(newValue)
                        if self.launchAgentWriteError != nil {
                            self.launchAgentWriteDisabled = GatewayLaunchAgentManager.isLaunchAgentWriteDisabled()
                            return
                        }
                        if newValue {
                            Task {
                                _ = await GatewayLaunchAgentManager.set(
                                    enabled: false,
                                    bundlePath: Bundle.main.bundlePath,
                                    port: GatewayEnvironment.gatewayPort())
                            }
                        }
                    }

                Text(
                    "When enabled, Marv won't install or manage \(gatewayLaunchdLabel). " +
                        "It will only attach to an existing Gateway.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if let launchAgentWriteError {
                    Text(launchAgentWriteError)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Debug")
                .font(.title3.weight(.semibold))
            Text("Tools for diagnosing local issues (Gateway, ports, logs).")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    private func gridLabel(_ text: String) -> some View {
        Text(text)
            .foregroundStyle(.secondary)
            .frame(width: self.labelColumnWidth, alignment: .leading)
    }

    private var appInfoSection: some View {
        GroupBox("App") {
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 10) {
                GridRow {
                    self.gridLabel("Health")
                    HStack(spacing: 8) {
                        Circle().fill(self.healthStore.state.tint).frame(width: 10, height: 10)
                        Text(self.healthStore.summaryLine)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                GridRow {
                    self.gridLabel("CLI")
                    let loc = CLIInstaller.installedLocation()
                    Text(loc ?? "missing")
                        .font(.caption.monospaced())
                        .foregroundStyle(loc == nil ? Color.red : Color.secondary)
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                GridRow {
                    self.gridLabel("PID")
                    Text("\(ProcessInfo.processInfo.processIdentifier)")
                }
                GridRow {
                    self.gridLabel("Binary path")
                    Text(Bundle.main.bundlePath)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
        }
    }

    private var gatewaySection: some View {
        GroupBox("Gateway") {
            VStack(alignment: .leading, spacing: 10) {
                Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 10) {
                    GridRow {
                        self.gridLabel("Status")
                        HStack(spacing: 8) {
                            Text(self.gatewayManager.status.label)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }

                let key = DeepLinkHandler.currentKey()
                HStack(spacing: 8) {
                    Text("Key")
                        .foregroundStyle(.secondary)
                        .frame(width: self.labelColumnWidth, alignment: .leading)
                    Text(key)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Button("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(key, forType: .string)
                    }
                    .buttonStyle(.bordered)
                    Button("Copy sample URL") {
                        let msg = "Hello from deep link"
                        let encoded = msg.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? msg
                        let url = "marv://agent?message=\(encoded)&key=\(key)"
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(url, forType: .string)
                    }
                    .buttonStyle(.bordered)
                    Spacer(minLength: 0)
                }

                Text("Deep links (marv://…) are always enabled; the key controls unattended runs.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 6) {
                    Text("Stdout / stderr")
                        .font(.caption.weight(.semibold))
                    ScrollView {
                        Text(self.gatewayManager.log.isEmpty ? "—" : self.gatewayManager.log)
                            .font(.caption.monospaced())
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(height: 180)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.2)))

                    HStack(spacing: 8) {
                        if self.canRestartGateway {
                            Button("Restart Gateway") { DebugActions.restartGateway() }
                        }
                        Button("Clear log") { GatewayProcessManager.shared.clearLog() }
                        Spacer(minLength: 0)
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
    }

    private var logsSection: some View {
        GroupBox("Logs") {
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 10) {
                GridRow {
                    self.gridLabel("Pino log")
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            Button("Open") { DebugActions.openLog() }
                                .buttonStyle(.bordered)
                            Text(DebugActions.pinoLogPath())
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                }

                GridRow {
                    self.gridLabel("App logging")
                    VStack(alignment: .leading, spacing: 8) {
                        Picker("Verbosity", selection: self.$appLogLevelRaw) {
                            ForEach(AppLogLevel.allCases) { level in
                                Text(level.title).tag(level.rawValue)
                            }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                        .help("Controls the macOS app log verbosity.")

                        Toggle("Write rolling diagnostics log (JSONL)", isOn: self.$diagnosticsFileLogEnabled)
                            .toggleStyle(.checkbox)
                            .help(
                                "Writes a rotating, local-only log under ~/Library/Logs/Marv/. " +
                                    "Enable only while actively debugging.")

                        HStack(spacing: 8) {
                            Button("Open folder") {
                                NSWorkspace.shared.open(DiagnosticsFileLog.logDirectoryURL())
                            }
                            .buttonStyle(.bordered)
                            Button("Clear") {
                                Task { try? await DiagnosticsFileLog.shared.clear() }
                            }
                            .buttonStyle(.bordered)
                        }
                        Text(DiagnosticsFileLog.logFileURL().path)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
            }
        }
    }

    private var portsSection: some View {
        GroupBox("Ports") {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Text("Port diagnostics")
                        .font(.caption.weight(.semibold))
                    if self.portCheckInFlight { ProgressView().controlSize(.small) }
                    Spacer()
                    Button("Check gateway ports") {
                        Task { await self.runPortCheck() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(self.portCheckInFlight)
                    Button("Reset SSH tunnel") {
                        Task { await self.resetGatewayTunnel() }
                    }
                    .buttonStyle(.bordered)
                    .disabled(self.tunnelResetInFlight || !self.isRemoteMode)
                }

                if let portKillStatus {
                    Text(portKillStatus)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let tunnelResetStatus {
                    Text(tunnelResetStatus)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if self.portReports.isEmpty, !self.portCheckInFlight {
                    Text("Check which process owns \(GatewayEnvironment.gatewayPort()) and suggest fixes.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(self.portReports) { report in
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Port \(report.port)")
                                .font(.footnote.weight(.semibold))
                            Text(report.summary)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                            ForEach(report.listeners) { listener in
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack(spacing: 8) {
                                        Text("\(listener.command) (\(listener.pid))")
                                            .font(.caption.monospaced())
                                            .foregroundStyle(listener.expected ? .secondary : Color.red)
                                            .lineLimit(1)
                                        Spacer()
                                        Button("Kill") {
                                            self.requestKill(listener)
                                        }
                                        .buttonStyle(.bordered)
                                    }
                                    Text(listener.fullCommand)
                                        .font(.caption2.monospaced())
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                        .truncationMode(.middle)
                                }
                                .padding(6)
                                .background(Color.secondary.opacity(0.05))
                                .cornerRadius(4)
                            }
                        }
                        .padding(8)
                        .background(Color.secondary.opacity(0.08))
                        .cornerRadius(6)
                    }
                }
            }
        }
    }

    private var pathsSection: some View {
        GroupBox("Paths") {
            VStack(alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Marv project root")
                        .font(.caption.weight(.semibold))
                    HStack(spacing: 8) {
                        TextField("Path to marv repo", text: self.$gatewayRootInput)
                            .textFieldStyle(.roundedBorder)
                            .font(.caption.monospaced())
                            .onSubmit { self.saveRelayRoot() }
                        Button("Save") { self.saveRelayRoot() }
                            .buttonStyle(.borderedProminent)
                        Button("Reset") {
                            let def = FileManager().homeDirectoryForCurrentUser
                                .appendingPathComponent("Projects/marv").path
                            self.gatewayRootInput = def
                            self.saveRelayRoot()
                        }
                        .buttonStyle(.bordered)
                    }
                    Text("Used for pnpm/node fallback and PATH population when launching the gateway.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Divider()

                Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 10) {
                    GridRow {
                        self.gridLabel("Session store")
                        VStack(alignment: .leading, spacing: 6) {
                            HStack(spacing: 8) {
                                TextField("Path", text: self.$sessionStorePath)
                                    .textFieldStyle(.roundedBorder)
                                    .font(.caption.monospaced())
                                    .frame(width: 360)
                                Button("Save") { self.saveSessionStorePath() }
                                    .buttonStyle(.borderedProminent)
                            }
                            if let sessionStoreSaveError {
                                Text(sessionStoreSaveError)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            } else {
                                Text("Used by the CLI session loader; stored in ~/.marv/marv.json.")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    GridRow {
                        self.gridLabel("Model catalog")
                        VStack(alignment: .leading, spacing: 6) {
                            Text(self.modelCatalogPath)
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                            HStack(spacing: 8) {
                                Button {
                                    self.chooseCatalogFile()
                                } label: {
                                    Label("Choose models.generated.ts…", systemImage: "folder")
                                }
                                .buttonStyle(.bordered)

                                Button {
                                    Task { await self.reloadModels() }
                                } label: {
                                    Label(
                                        self.modelsLoading ? "Reloading…" : "Reload models",
                                        systemImage: "arrow.clockwise")
                                }
                                .buttonStyle(.bordered)
                                .disabled(self.modelsLoading)
                            }
                            if let modelsError {
                                Text(modelsError)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            } else if let modelsCount {
                                Text("Loaded \(modelsCount) models")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            Text("Local fallback for model picker when gateway models.list is unavailable.")
                                .font(.footnote)
                                .foregroundStyle(.tertiary)
                        }
                    }
                }
            }
        }
    }

    private var quickActionsSection: some View {
        GroupBox("Quick actions") {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Button("Send Test Notification") {
                        Task { await DebugActions.sendTestNotification() }
                    }
                    .buttonStyle(.bordered)

                    Spacer(minLength: 0)
                }

                VStack(alignment: .leading, spacing: 6) {
                    Text(
                        "Note: macOS may require restarting Marv after enabling Accessibility or Screen Recording.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    Button {
                        LaunchdManager.startMarv()
                    } label: {
                        Label("Restart Marv", systemImage: "arrow.counterclockwise")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                HStack(spacing: 8) {
                    Button("Restart app") { DebugActions.restartApp() }
                    Button("Restart onboarding") { DebugActions.restartOnboarding() }
                    Button("Reveal app in Finder") { self.revealApp() }
                    Spacer(minLength: 0)
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private var experimentsSection: some View {
        GroupBox("Experiments") {
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 10) {
                GridRow {
                    self.gridLabel("Icon override")
                    Picker("", selection: self.bindingOverride) {
                        ForEach(IconOverrideSelection.allCases) { option in
                            Text(option.label).tag(option.rawValue)
                        }
                    }
                    .labelsHidden()
                    .frame(maxWidth: 280, alignment: .leading)
                }
                GridRow {
                    self.gridLabel("Chat")
                    Text("Native SwiftUI")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @MainActor
    private func runPortCheck() async {
        self.portCheckInFlight = true
        self.portKillStatus = nil
        let reports = await DebugActions.checkGatewayPorts()
        self.portReports = reports
        self.portCheckInFlight = false
    }

    @MainActor
    private func resetGatewayTunnel() async {
        self.tunnelResetInFlight = true
        self.tunnelResetStatus = nil
        let result = await DebugActions.resetGatewayTunnel()
        switch result {
        case let .success(message):
            self.tunnelResetStatus = message
        case let .failure(err):
            self.tunnelResetStatus = err.localizedDescription
        }
        await self.runPortCheck()
        self.tunnelResetInFlight = false
    }

    @MainActor
    private func requestKill(_ listener: DebugActions.PortListener) {
        if listener.expected {
            self.pendingKill = listener
        } else {
            Task { await self.killConfirmed(listener.pid) }
        }
    }

    @MainActor
    private func killConfirmed(_ pid: Int32) async {
        let result = await DebugActions.killProcess(Int(pid))
        switch result {
        case .success:
            self.portKillStatus = "Sent kill to \(pid)."
            await self.runPortCheck()
        case let .failure(err):
            self.portKillStatus = "Kill \(pid) failed: \(err.localizedDescription)"
        }
    }

    private func chooseCatalogFile() {
        let panel = NSOpenPanel()
        panel.title = "Select models.generated.ts"
        let tsType = UTType(filenameExtension: "ts")
            ?? UTType(tag: "ts", tagClass: .filenameExtension, conformingTo: .sourceCode)
            ?? .item
        panel.allowedContentTypes = [tsType]
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: self.modelCatalogPath).deletingLastPathComponent()
        if panel.runModal() == .OK, let url = panel.url {
            self.modelCatalogPath = url.path
            self.modelCatalogReloadBump += 1
            Task { await self.reloadModels() }
        }
    }

    private func reloadModels() async {
        guard !self.modelsLoading else { return }
        self.modelsLoading = true
        self.modelsError = nil
        self.modelCatalogReloadBump += 1
        defer { self.modelsLoading = false }
        do {
            let loaded = try await ModelCatalogLoader.load(from: self.modelCatalogPath)
            self.modelsCount = loaded.count
        } catch {
            self.modelsCount = nil
            self.modelsError = error.localizedDescription
        }
    }

    private func revealApp() {
        let url = Bundle.main.bundleURL
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    private func saveRelayRoot() {
        GatewayProcessManager.shared.setProjectRoot(path: self.gatewayRootInput)
    }

    private func loadSessionStorePath() {
        let url = self.configURL()
        guard
            let data = try? Data(contentsOf: url),
            let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let session = parsed["session"] as? [String: Any],
            let path = session["store"] as? String
        else {
            self.sessionStorePath = SessionLoader.defaultStorePath
            return
        }
        self.sessionStorePath = path
    }

    private func saveSessionStorePath() {
        let trimmed = self.sessionStorePath.trimmingCharacters(in: .whitespacesAndNewlines)
        var root: [String: Any] = [:]
        let url = self.configURL()
        if let data = try? Data(contentsOf: url),
           let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            root = parsed
        }

        var session = root["session"] as? [String: Any] ?? [:]
        session["store"] = trimmed.isEmpty ? SessionLoader.defaultStorePath : trimmed
        root["session"] = session

        do {
            let data = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
            try FileManager().createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try data.write(to: url, options: [.atomic])
            self.sessionStoreSaveError = nil
        } catch {
            self.sessionStoreSaveError = error.localizedDescription
        }
    }

    private var bindingOverride: Binding<String> {
        Binding {
            self.iconOverrideRaw
        } set: { newValue in
            self.iconOverrideRaw = newValue
            if let selection = IconOverrideSelection(rawValue: newValue) {
                Task { @MainActor in
                    AppStateStore.shared.iconOverride = selection
                }
            }
        }
    }

    private var isRemoteMode: Bool {
        CommandResolver.connectionSettings().mode == .remote
    }

    private var canRestartGateway: Bool {
        self.state.connectionMode == .local
    }

    private func configURL() -> URL {
        MarvPaths.configURL
    }
}

struct PlainSettingsGroupBoxStyle: GroupBoxStyle {
    func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            configuration.label
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            configuration.content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

#if DEBUG
struct DebugSettings_Previews: PreviewProvider {
    static var previews: some View {
        DebugSettings(state: .preview)
            .frame(width: SettingsTab.windowWidth, height: SettingsTab.windowHeight)
    }
}

@MainActor
extension DebugSettings {
    static func exerciseForTesting() async {
        let view = DebugSettings(state: .preview)
        view.modelsCount = 3
        view.modelsLoading = false
        view.modelsError = "Failed to load models"
        view.gatewayRootInput = "/tmp/marv"
        view.sessionStorePath = "/tmp/sessions.json"
        view.sessionStoreSaveError = "Save failed"
        view.portCheckInFlight = true
        view.portReports = [
            DebugActions.PortReport(
                port: GatewayEnvironment.gatewayPort(),
                expected: "Gateway websocket (node/tsx)",
                status: .missing("Missing"),
                listeners: []),
        ]
        view.portKillStatus = "Killed"
        view.pendingKill = DebugActions.PortListener(
            pid: 1,
            command: "node",
            fullCommand: "node",
            user: nil,
            expected: true)
        _ = view.body
        _ = view.header
        _ = view.appInfoSection
        _ = view.gatewaySection
        _ = view.logsSection
        _ = view.portsSection
        _ = view.pathsSection
        _ = view.quickActionsSection
        _ = view.experimentsSection
        _ = view.gridLabel("Test")

        view.loadSessionStorePath()
        await view.reloadModels()
    }
}
#endif

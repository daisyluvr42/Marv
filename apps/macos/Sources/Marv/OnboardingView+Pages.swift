import AppKit
import MarvIPC
import SwiftUI

extension OnboardingView {
    @ViewBuilder
    func pageView(for pageIndex: Int) -> some View {
        switch pageIndex {
        case 0:
            self.welcomePage()
        case 1:
            self.providerPage()
        case 2:
            self.modelPage()
        case 3:
            self.readyPage()
        default:
            EmptyView()
        }
    }

    // MARK: - Page 0: Welcome

    func welcomePage() -> some View {
        self.onboardingPage {
            VStack(spacing: 22) {
                Text("Welcome to Marv")
                    .font(.largeTitle.weight(.semibold))
                Text("Your personal AI agent assistant. Let's set up your AI provider and model to get started.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
                    .frame(maxWidth: 560)
                    .fixedSize(horizontal: false, vertical: true)

                self.onboardingCard(spacing: 10, padding: 14) {
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(Color(nsColor: .systemOrange))
                            .frame(width: 22)
                            .padding(.top, 1)

                        VStack(alignment: .leading, spacing: 6) {
                            Text("Security notice")
                                .font(.headline)
                            Text(
                                "The connected AI agent (e.g. Claude) can trigger powerful actions on your Mac, " +
                                    "including running commands, reading/writing files, and capturing screenshots — " +
                                    "depending on the permissions you grant.\n\n" +
                                    "Only enable Marv if you understand the risks and trust the prompts and " +
                                    "integrations you use.")
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                .frame(maxWidth: 520)
            }
            .padding(.top, 16)
        }
    }

    // MARK: - Page 1: Provider & API Key

    func providerPage() -> some View {
        self.onboardingPage {
            Text("Choose AI Provider")
                .font(.largeTitle.weight(.semibold))
            Text("Select your AI provider and enter your API key.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 520)
                .fixedSize(horizontal: false, vertical: true)

            self.onboardingCard(spacing: 12, padding: 14) {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(Self.providerDisplayNames, id: \.id) { provider in
                        self.providerChoiceButton(
                            title: provider.name,
                            systemImage: provider.icon,
                            selected: self.selectedProvider == provider.id)
                        {
                            withAnimation(.spring(response: 0.25, dampingFraction: 0.9)) {
                                self.selectedProvider = provider.id
                            }
                        }
                    }

                    Divider().padding(.vertical, 4)

                    if self.selectedProvider == "custom" {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Base URL")
                                .font(.callout.weight(.semibold))
                            TextField("https://api.example.com/v1", text: self.$customBaseUrl)
                                .textFieldStyle(.roundedBorder)

                            Text("API Compatibility")
                                .font(.callout.weight(.semibold))
                            Picker("API Compatibility", selection: self.$customApiCompatibility) {
                                Text("OpenAI-compatible").tag("openai")
                                Text("Anthropic-compatible").tag("anthropic")
                            }
                            .pickerStyle(.segmented)

                            Text("Model ID")
                                .font(.callout.weight(.semibold))
                            TextField("model-name", text: self.$customModelId)
                                .textFieldStyle(.roundedBorder)
                        }
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text("API Key")
                            .font(.callout.weight(.semibold))
                        SecureField(self.apiKeyPlaceholder, text: self.$apiKey)
                            .textFieldStyle(.roundedBorder)

                        Text(self.apiKeyHint)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var apiKeyPlaceholder: String {
        switch self.selectedProvider {
        case "anthropic": return "sk-ant-..."
        case "openai": return "sk-..."
        case "google": return "AIza..."
        case "moonshot": return "sk-..."
        default: return "API key"
        }
    }

    private var apiKeyHint: String {
        switch self.selectedProvider {
        case "anthropic":
            return "Get your API key at console.anthropic.com"
        case "openai":
            return "Get your API key at platform.openai.com"
        case "google":
            return "Get your API key at aistudio.google.com"
        case "moonshot":
            return "Get your API key at platform.moonshot.cn"
        default:
            return "Enter the API key for your custom provider."
        }
    }

    private func providerChoiceButton(
        title: String,
        systemImage: String,
        selected: Bool,
        action: @escaping () -> Void) -> some View
    {
        Button {
            action()
        } label: {
            HStack(alignment: .center, spacing: 10) {
                Image(systemName: systemImage)
                    .font(.body.weight(.semibold))
                    .foregroundStyle(selected ? Color.accentColor : .secondary)
                    .frame(width: 22)
                Text(title)
                    .font(.callout.weight(.semibold))
                    .lineLimit(1)
                Spacer(minLength: 0)
                if selected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Color.accentColor)
                } else {
                    Image(systemName: "circle")
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(selected ? Color.accentColor.opacity(0.12) : Color.clear))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(
                        selected ? Color.accentColor.opacity(0.45) : Color.clear,
                        lineWidth: 1))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Page 2: Model Selection

    func modelPage() -> some View {
        self.onboardingPage {
            Text("Choose Default Model")
                .font(.largeTitle.weight(.semibold))
            Text("Select the model to use by default. You can change this later in Settings.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 520)
                .fixedSize(horizontal: false, vertical: true)

            self.onboardingCard(spacing: 12, padding: 14) {
                if self.selectedProvider == "custom" {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Model ID")
                            .font(.callout.weight(.semibold))
                        TextField("model-name", text: self.$selectedModel)
                            .textFieldStyle(.roundedBorder)
                        Text("Enter the model identifier for your custom provider.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else if let models = Self.providerModels[self.selectedProvider] {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(models, id: \.id) { model in
                            self.modelChoiceButton(
                                title: model.name,
                                modelId: model.id,
                                selected: self.selectedModel == model.id)
                            {
                                withAnimation(.spring(response: 0.25, dampingFraction: 0.9)) {
                                    self.selectedModel = model.id
                                }
                            }
                        }
                    }
                } else {
                    Text("No models available for this provider.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func modelChoiceButton(
        title: String,
        modelId: String,
        selected: Bool,
        action: @escaping () -> Void) -> some View
    {
        Button {
            action()
        } label: {
            HStack(alignment: .center, spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.callout.weight(.semibold))
                        .lineLimit(1)
                    Text(modelId)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
                if selected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Color.accentColor)
                } else {
                    Image(systemName: "circle")
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(selected ? Color.accentColor.opacity(0.12) : Color.clear))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(
                        selected ? Color.accentColor.opacity(0.45) : Color.clear,
                        lineWidth: 1))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Page 3: Setup

    private static let cliInstallCommand = "npm install -g agentmarv@latest && marv gateway run"

    func readyPage() -> some View {
        self.onboardingPage {
            if self.setupPhase == .done {
                self.setupSuccessContent()
            } else if self.isSetupFailed {
                self.setupFailedContent()
            } else {
                self.setupProgressContent()
            }
        }
    }

    private func setupProgressContent() -> some View {
        VStack(spacing: 22) {
            Text("Setting Up...")
                .font(.largeTitle.weight(.semibold))
            Text("Configuring Marv on your Mac. This may take a minute.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 520)

            self.onboardingCard(spacing: 16, padding: 20) {
                self.setupStepRow(
                    title: "Save configuration",
                    phase: .savingConfig)
                Divider()
                self.setupStepRow(
                    title: "Check environment",
                    phase: .checkingEnvironment)
                Divider()
                self.setupStepRow(
                    title: "Install CLI",
                    phase: .installingCLI)
                Divider()
                self.setupStepRow(
                    title: "Start gateway",
                    phase: .startingGateway)
            }

            if !self.setupDetailMessage.isEmpty {
                Text(self.setupDetailMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 480)
                    .transition(.opacity)
            }
        }
        .padding(.top, 8)
    }

    private func setupStepRow(title: String, phase: SetupPhase) -> some View {
        let stepState = self.stepState(for: phase)
        return HStack(spacing: 12) {
            Group {
                switch stepState {
                case .pending:
                    Image(systemName: "circle")
                        .foregroundStyle(.tertiary)
                case .active:
                    ProgressView()
                        .controlSize(.small)
                case .done:
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                case .skipped:
                    Image(systemName: "arrow.right.circle.fill")
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: 20, height: 20)

            Text(title)
                .font(.callout.weight(stepState == .active ? .semibold : .regular))
                .foregroundStyle(stepState == .pending ? .secondary : .primary)

            Spacer()
        }
        .padding(.vertical, 2)
    }

    private enum StepState { case pending, active, done, skipped }

    private func stepState(for phase: SetupPhase) -> StepState {
        let order: [SetupPhase] = [.savingConfig, .checkingEnvironment, .installingCLI, .startingGateway]
        guard let targetIdx = order.firstIndex(of: phase) else { return .pending }
        guard let currentIdx = order.firstIndex(of: self.setupPhase) else {
            // .done or .failed — all steps are done
            if self.setupPhase == .done || self.isSetupFailed {
                return .done
            }
            return .pending
        }
        if currentIdx > targetIdx { return .done }
        if currentIdx == targetIdx { return .active }
        return .pending
    }

    private func setupSuccessContent() -> some View {
        VStack(spacing: 22) {
            Text("All Set!")
                .font(.largeTitle.weight(.semibold))

            self.onboardingCard {
                self.featureRow(
                    title: "Provider",
                    subtitle: self.providerDisplayName,
                    systemImage: "cpu")
                Divider().padding(.vertical, 2)
                self.featureRow(
                    title: "Model",
                    subtitle: self.selectedModel.isEmpty ? "Default" : self.selectedModel,
                    systemImage: "brain")
                Divider().padding(.vertical, 2)
                self.featureRow(
                    title: "Gateway",
                    subtitle: "Running (port \(GatewayEnvironment.gatewayPort()))",
                    systemImage: "network")
                Divider().padding(.vertical, 6)
                Toggle("Launch at login", isOn: self.$state.launchAtLogin)
                    .onChange(of: self.state.launchAtLogin) { _, newValue in
                        AppStateStore.updateLaunchAtLogin(enabled: newValue)
                    }
            }
        }
        .padding(.top, 8)
    }

    private func setupFailedContent() -> some View {
        VStack(spacing: 22) {
            Text("Setup Issue")
                .font(.largeTitle.weight(.semibold))

            if case let .failed(reason) = self.setupPhase {
                self.onboardingCard(spacing: 10, padding: 14) {
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(Color(nsColor: .systemOrange))
                            .frame(width: 22)
                            .padding(.top, 1)

                        VStack(alignment: .leading, spacing: 6) {
                            Text(reason)
                                .font(.subheadline)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }

            self.onboardingCard(spacing: 10, padding: 14) {
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: "terminal.fill")
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 22)
                        .padding(.top, 1)

                    VStack(alignment: .leading, spacing: 8) {
                        Text("Manual Setup")
                            .font(.headline)
                        Text("You can also install manually in Terminal:")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)

                        HStack(spacing: 8) {
                            Text(Self.cliInstallCommand)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(.primary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 8)
                                .background(
                                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                                        .fill(Color.primary.opacity(0.06)))

                            Button {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(Self.cliInstallCommand, forType: .string)
                            } label: {
                                Image(systemName: "doc.on.doc")
                                    .font(.body.weight(.medium))
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .help("Copy to clipboard")
                        }
                    }
                }
            }

            Button("Skip for now") {
                self.skipSetup()
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .font(.callout)
        }
        .padding(.top, 8)
    }

    var providerDisplayName: String {
        Self.providerDisplayNames.first(where: { $0.id == self.selectedProvider })?.name
            ?? self.selectedProvider
    }
}

import AppKit
import Combine
import Observation
import MarvIPC
import SwiftUI

enum UIStrings {
    static let welcomeTitle = "Welcome to Marv"
}

@MainActor
final class OnboardingController {
    static let shared = OnboardingController()
    private var window: NSWindow?

    func show() {
        if ProcessInfo.processInfo.isNixMode {
            // Nix mode is fully declarative; onboarding would suggest interactive setup that doesn't apply.
            UserDefaults.standard.set(true, forKey: "marv.onboardingSeen")
            UserDefaults.standard.set(currentOnboardingVersion, forKey: onboardingVersionKey)
            AppStateStore.shared.onboardingSeen = true
            return
        }
        if let window {
            DockIconManager.shared.temporarilyShowDock()
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        let hosting = NSHostingController(rootView: OnboardingView())
        let window = NSWindow(contentViewController: hosting)
        window.title = UIStrings.welcomeTitle
        window.setContentSize(NSSize(width: OnboardingView.windowWidth, height: OnboardingView.windowHeight))
        window.styleMask = [.titled, .closable, .fullSizeContentView]
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.isMovableByWindowBackground = true
        window.center()
        DockIconManager.shared.temporarilyShowDock()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        self.window = window
    }

    func close() {
        self.window?.close()
        self.window = nil
    }

    func restart() {
        self.close()
        self.show()
    }
}

struct OnboardingView: View {
    @Environment(\.openSettings) var openSettings
    @State var currentPage = 0
    @State var isRequesting = false
    @State var workspacePath: String = ""
    @State var workspaceStatus: String?
    @State var workspaceApplying = false
    @Bindable var state: AppState

    // Provider & API key state
    @State var selectedProvider: String = "anthropic"
    @State var apiKey: String = ""
    @State var customBaseUrl: String = ""
    @State var customModelId: String = ""
    @State var selectedModel: String = ""
    @State var apiKeyValid: Bool = false
    @State var savingConfig: Bool = false

    static let windowWidth: CGFloat = 630
    static let windowHeight: CGFloat = 580

    let pageWidth: CGFloat = Self.windowWidth
    let contentHeight: CGFloat = 400

    var totalPages: Int { 4 }

    var pageOrder: [Int] { [0, 1, 2, 3] }

    var pageCount: Int {
        self.pageOrder.count
    }

    var activePageIndex: Int {
        self.activePageIndex(for: self.currentPage)
    }

    var buttonTitle: String {
        self.currentPage == self.pageCount - 1 ? "Finish" : "Next"
    }

    var canAdvance: Bool {
        switch self.activePageIndex {
        case 1:
            // Provider page: require API key (or custom base URL)
            if self.selectedProvider == "custom" {
                return !self.customBaseUrl.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    && !self.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
            return !self.apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        default:
            return true
        }
    }

    static let providerDisplayNames: [(id: String, name: String, icon: String)] = [
        ("anthropic", "Anthropic (Claude)", "brain.head.profile"),
        ("openai", "OpenAI", "sparkle"),
        ("google", "Google (Gemini)", "globe"),
        ("moonshot", "Moonshot", "moon.stars"),
        ("custom", "Custom", "wrench.and.screwdriver"),
    ]

    static let providerModels: [String: [(id: String, name: String)]] = [
        "anthropic": [
            ("claude-opus-4-6", "Claude Opus 4.6"),
            ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
            ("claude-haiku-3.5", "Claude Haiku 3.5"),
        ],
        "openai": [
            ("gpt-5", "GPT-5"),
            ("gpt-5-mini", "GPT-5 Mini"),
            ("gpt-4.1", "GPT-4.1"),
        ],
        "google": [
            ("gemini-2.5-pro", "Gemini 2.5 Pro"),
            ("gemini-2.5-flash", "Gemini 2.5 Flash"),
        ],
        "moonshot": [
            ("moonshot-v1-128k", "Moonshot v1 128K"),
            ("moonshot-v1-32k", "Moonshot v1 32K"),
        ],
    ]

    init(
        state: AppState = AppStateStore.shared)
    {
        self.state = state
    }
}

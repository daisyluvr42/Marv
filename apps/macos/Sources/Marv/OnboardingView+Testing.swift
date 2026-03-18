import SwiftUI

#if DEBUG
@MainActor
extension OnboardingView {
    static func exerciseForTesting() {
        let state = AppState(preview: true)
        let view = OnboardingView(state: state)

        // Exercise provider/model state
        var mutableView = view
        mutableView.selectedProvider = "anthropic"
        mutableView.apiKey = "sk-ant-test"
        mutableView.selectedModel = "claude-opus-4-6"
        mutableView.customBaseUrl = "https://api.example.com"
        mutableView.customModelId = "custom-model"

        // Exercise all pages
        _ = mutableView.welcomePage()
        _ = mutableView.providerPage()
        _ = mutableView.modelPage()
        _ = mutableView.readyPage()

        // Exercise navigation
        mutableView.currentPage = 0
        mutableView.handleNext()
        mutableView.handleBack()

        // Exercise layout helpers
        _ = mutableView.onboardingPage { Text("Test") }
        _ = mutableView.onboardingCard { Text("Card") }
        _ = mutableView.featureRow(title: "Feature", subtitle: "Subtitle", systemImage: "sparkles")
        _ = mutableView.featureActionRow(
            title: "Action",
            subtitle: "Action subtitle",
            systemImage: "gearshape",
            buttonTitle: "Action",
            action: {})

        // Exercise provider switching
        mutableView.selectedProvider = "openai"
        mutableView.updateDefaultModel()
        _ = mutableView.providerPage()
        _ = mutableView.modelPage()

        mutableView.selectedProvider = "custom"
        mutableView.updateDefaultModel()
        _ = mutableView.providerPage()
        _ = mutableView.modelPage()

        mutableView.selectedProvider = "google"
        mutableView.updateDefaultModel()
        _ = mutableView.modelPage()

        mutableView.selectedProvider = "moonshot"
        mutableView.updateDefaultModel()
        _ = mutableView.modelPage()
    }
}
#endif

import Foundation
import SwiftUI
import Testing
@testable import Marv

@Suite(.serialized)
@MainActor
struct OnboardingViewSmokeTests {
    @Test func onboardingViewBuildsBody() {
        let state = AppState(preview: true)
        let view = OnboardingView(state: state)
        _ = view.body
    }

    @Test func pageOrderIsFixedFourPages() {
        let state = AppState(preview: true)
        let view = OnboardingView(state: state)
        #expect(view.pageOrder == [0, 1, 2, 3])
        #expect(view.totalPages == 4)
        #expect(view.pageCount == 4)
    }

    @Test func canAdvanceRequiresApiKey() {
        let state = AppState(preview: true)
        let view = OnboardingView(state: state)
        view.currentPage = 1
        view.selectedProvider = "anthropic"
        view.apiKey = ""
        #expect(!view.canAdvance)
        view.apiKey = "sk-ant-test"
        #expect(view.canAdvance)
    }

    @Test func customProviderRequiresBaseUrlApiKeyAndModel() {
        let state = AppState(preview: true)
        let view = OnboardingView(state: state)
        view.currentPage = 1
        view.selectedProvider = "custom"
        view.customBaseUrl = ""
        view.customModelId = ""
        view.apiKey = ""
        #expect(!view.canAdvance)

        view.customBaseUrl = "https://models.custom.local/v1"
        #expect(!view.canAdvance)

        view.apiKey = "custom-test-key"
        #expect(!view.canAdvance)

        view.customModelId = "local-large"
        #expect(view.canAdvance)
    }

    @Test func customProviderPersistsValidConfigAndAuthProfile() async throws {
        let tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("marv-onboarding-\(UUID().uuidString)", isDirectory: true)
        let stateDir = tempRoot.appendingPathComponent("state", isDirectory: true)
        let configPath = stateDir.appendingPathComponent("marv.json")
        let agentDir = tempRoot.appendingPathComponent("agents/main/agent", isDirectory: true)

        try FileManager.default.createDirectory(at: stateDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: agentDir, withIntermediateDirectories: true)

        try await TestIsolation.withIsolatedState(env: [
            "MARV_STATE_DIR": stateDir.path,
            "MARV_CONFIG_PATH": configPath.path,
            "MARV_AGENT_DIR": agentDir.path,
            "PI_CODING_AGENT_DIR": agentDir.path,
        ]) {
            await ConfigStore._testSetOverrides(.init(
                isRemoteMode: { false },
                loadLocal: { [:] },
                saveLocal: { root in
                    MarvConfigFile.saveDict(root)
                }))

            do {
                let state = AppState(preview: true)
                let view = OnboardingView(state: state)
                view.selectedProvider = "custom"
                view.customBaseUrl = "https://models.custom.local/v1"
                view.customModelId = "local-large"
                view.selectedModel = "local-large"
                view.customApiCompatibility = "anthropic"
                view.apiKey = "custom-test-key"

                view._testWriteApiKeyAuthProfile()
                await view._testSaveProviderAndModelConfig()

                let authData = try Data(contentsOf: MarvPaths.authProfilesURL)
                let authRoot = try #require(
                    JSONSerialization.jsonObject(with: authData) as? [String: Any])
                let profiles = try #require(authRoot["profiles"] as? [String: Any])
                let profileId = view._testAuthProfileId()
                let profile = try #require(profiles[profileId] as? [String: Any])
                #expect(profile["provider"] as? String == "custom-models-custom-local")
                #expect(profile["type"] as? String == "api_key")
                #expect(profile["key"] as? String == "custom-test-key")

                let savedRoot = MarvConfigFile.loadDict()
                let agents = try #require(savedRoot["agents"] as? [String: Any])
                let defaults = try #require(agents["defaults"] as? [String: Any])
                let model = try #require(defaults["model"] as? [String: Any])
                #expect(model["primary"] as? String == "custom-models-custom-local/local-large")
                #expect(defaults["modelPool"] == nil)

                let auth = try #require(savedRoot["auth"] as? [String: Any])
                let authProfiles = try #require(auth["profiles"] as? [String: Any])
                let authProfile = try #require(authProfiles[profileId] as? [String: Any])
                #expect(authProfile["provider"] as? String == "custom-models-custom-local")
                #expect(authProfile["mode"] as? String == "api_key")

                let models = try #require(savedRoot["models"] as? [String: Any])
                let providers = try #require(models["providers"] as? [String: Any])
                let provider = try #require(providers["custom-models-custom-local"] as? [String: Any])
                #expect(provider["baseUrl"] as? String == "https://models.custom.local/v1")
                #expect(provider["api"] as? String == "anthropic-messages")
                let configuredModels = try #require(provider["models"] as? [[String: Any]])
                #expect(configuredModels.first?["id"] as? String == "local-large")

                await ConfigStore._testClearOverrides()
            } catch {
                await ConfigStore._testClearOverrides()
                throw error
            }
        }
    }
}

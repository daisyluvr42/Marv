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
        var view = OnboardingView(state: state)
        view.currentPage = 1
        view.selectedProvider = "anthropic"
        view.apiKey = ""
        #expect(!view.canAdvance)
        view.apiKey = "sk-ant-test"
        #expect(view.canAdvance)
    }
}

import SwiftUI
import Testing
@testable import Marv

@Suite(.serialized)
@MainActor
struct MenuContentSmokeTests {
    @Test func menuContentBuildsBodyLocalMode() {
        let state = AppState(preview: true)
        state.connectionMode = .local
        let view = MenuContent(state: state, updater: nil)
        _ = view.body
    }

    @Test func menuContentBuildsBodyRemoteMode() {
        let state = AppState(preview: true)
        state.connectionMode = .remote
        let view = MenuContent(state: state, updater: nil)
        _ = view.body
    }

    @Test func menuContentBuildsBodyUnconfiguredMode() {
        let state = AppState(preview: true)
        state.connectionMode = .unconfigured
        let view = MenuContent(state: state, updater: nil)
        _ = view.body
    }

    @Test func menuContentBuildsBodyWithDebugControls() {
        let state = AppState(preview: true)
        state.connectionMode = .local
        state.debugPaneEnabled = true
        state.heartbeatsEnabled = true
        let view = MenuContent(state: state, updater: nil)
        _ = view.body
    }
}

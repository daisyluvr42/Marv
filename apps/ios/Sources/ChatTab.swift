import OpenClawChatUI
import OpenClawKit
import SwiftUI

struct ChatTab: View {
    @Environment(CompanionAppModel.self) private var appModel

    var body: some View {
        NavigationStack {
            if self.appModel.isOperatorConnected {
                ChatTabHost(gateway: self.appModel.operatorGateway, sessionKey: self.appModel.sessionKey)
                    .id(self.appModel.sessionKey)
                    .navigationTitle("Chat")
                    .navigationBarTitleDisplayMode(.inline)
            } else {
                ContentUnavailableView(
                    "Gateway Not Connected",
                    systemImage: "bolt.horizontal.circle",
                    description: Text("Connect the iPhone companion in Settings, then your gateway session will show up here."))
            }
        }
    }
}

private struct ChatTabHost: View {
    @State private var viewModel: OpenClawChatViewModel

    init(gateway: GatewayNodeSession, sessionKey: String) {
        self._viewModel = State(
            initialValue: OpenClawChatViewModel(
                sessionKey: sessionKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "main" : sessionKey,
                transport: CompanionChatTransport(gateway: gateway)))
    }

    var body: some View {
        OpenClawChatView(viewModel: self.viewModel, showsSessionSwitcher: true)
    }
}

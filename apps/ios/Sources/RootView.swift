import SwiftUI

struct RootView: View {
    var body: some View {
        TabView {
            ChatTab()
                .tabItem { Label("Chat", systemImage: "message") }

            VoiceInputTab()
                .tabItem { Label("Voice", systemImage: "mic") }

            DashboardTab()
                .tabItem { Label("Dashboard", systemImage: "square.grid.2x2") }

            OperationsTab()
                .tabItem { Label("Operations", systemImage: "chart.bar") }

            SettingsTab()
                .tabItem { Label("Settings", systemImage: "gearshape") }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
    }
}

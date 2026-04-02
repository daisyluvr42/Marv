import SwiftUI

@main
struct MarvCompanionApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @State private var appModel = CompanionAppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(.systemBackground))
                .environment(self.appModel)
                .environment(self.appModel.speechInput)
                .task {
                    await self.appModel.loadAndConnectIfPossible()
                }
                .onChange(of: self.scenePhase) { _, newPhase in
                    self.appModel.setScenePhase(newPhase)
                    if newPhase == .active {
                        Task {
                            await self.appModel.handleSceneDidBecomeActive()
                        }
                    }
                }
        }
    }
}

import Foundation
import MarvIPC

extension OnboardingView {
    @MainActor
    func refreshPerms() async {
        // No-op: simplified onboarding does not manage permissions.
    }

    @MainActor
    func request(_ cap: Capability) async {
        // No-op: permissions are handled post-onboarding in Settings.
    }
}

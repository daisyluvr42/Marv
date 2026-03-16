import Foundation

// Stable identifier used for both the macOS LaunchAgent label and Nix-managed defaults suite.
// nix-marv writes app defaults into this suite to survive app bundle identifier churn.
let launchdLabel = "bot.marv.mac"
let gatewayLaunchdLabel = "bot.marv.gateway"
let onboardingVersionKey = "marv.onboardingVersion"
let onboardingSeenKey = "marv.onboardingSeen"
let currentOnboardingVersion = 7
let pauseDefaultsKey = "marv.pauseEnabled"
let iconAnimationsEnabledKey = "marv.iconAnimationsEnabled"
let showDockIconKey = "marv.showDockIcon"
let iconOverrideKey = "marv.iconOverride"
let connectionModeKey = "marv.connectionMode"
let remoteTargetKey = "marv.remoteTarget"
let remoteIdentityKey = "marv.remoteIdentity"
let remoteProjectRootKey = "marv.remoteProjectRoot"
let remoteCliPathKey = "marv.remoteCliPath"
let deepLinkKeyKey = "marv.deepLinkKey"
let modelCatalogPathKey = "marv.modelCatalogPath"
let modelCatalogReloadKey = "marv.modelCatalogReload"
let cliInstallPromptedVersionKey = "marv.cliInstallPromptedVersion"
let heartbeatsEnabledKey = "marv.heartbeatsEnabled"
let debugPaneEnabledKey = "marv.debugPaneEnabled"
let debugFileLogEnabledKey = "marv.debug.fileLogEnabled"
let appLogLevelKey = "marv.debug.appLogLevel"

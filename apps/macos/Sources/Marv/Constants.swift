import Foundation

// Stable identifier used for both the macOS LaunchAgent label and Nix-managed defaults suite.
// nix-marv writes app defaults into this suite to survive app bundle identifier churn.
let launchdLabel = "ai.marv.mac"
let gatewayLaunchdLabel = "ai.marv.gateway"
let onboardingVersionKey = "marv.onboardingVersion"
let onboardingSeenKey = "marv.onboardingSeen"
let currentOnboardingVersion = 7
let pauseDefaultsKey = "marv.pauseEnabled"
let iconAnimationsEnabledKey = "marv.iconAnimationsEnabled"
let swabbleEnabledKey = "marv.swabbleEnabled"
let swabbleTriggersKey = "marv.swabbleTriggers"
let voiceWakeTriggerChimeKey = "marv.voiceWakeTriggerChime"
let voiceWakeSendChimeKey = "marv.voiceWakeSendChime"
let showDockIconKey = "marv.showDockIcon"
let defaultVoiceWakeTriggers = ["marv"]
let voiceWakeMaxWords = 32
let voiceWakeMaxWordLength = 64
let voiceWakeMicKey = "marv.voiceWakeMicID"
let voiceWakeMicNameKey = "marv.voiceWakeMicName"
let voiceWakeLocaleKey = "marv.voiceWakeLocaleID"
let voiceWakeAdditionalLocalesKey = "marv.voiceWakeAdditionalLocaleIDs"
let voicePushToTalkEnabledKey = "marv.voicePushToTalkEnabled"
let talkEnabledKey = "marv.talkEnabled"
let iconOverrideKey = "marv.iconOverride"
let connectionModeKey = "marv.connectionMode"
let remoteTargetKey = "marv.remoteTarget"
let remoteIdentityKey = "marv.remoteIdentity"
let remoteProjectRootKey = "marv.remoteProjectRoot"
let remoteCliPathKey = "marv.remoteCliPath"
let canvasEnabledKey = "marv.canvasEnabled"
let cameraEnabledKey = "marv.cameraEnabled"
let systemRunPolicyKey = "marv.systemRunPolicy"
let systemRunAllowlistKey = "marv.systemRunAllowlist"
let systemRunEnabledKey = "marv.systemRunEnabled"
let locationModeKey = "marv.locationMode"
let locationPreciseKey = "marv.locationPreciseEnabled"
let peekabooBridgeEnabledKey = "marv.peekabooBridgeEnabled"
let deepLinkKeyKey = "marv.deepLinkKey"
let modelCatalogPathKey = "marv.modelCatalogPath"
let modelCatalogReloadKey = "marv.modelCatalogReload"
let cliInstallPromptedVersionKey = "marv.cliInstallPromptedVersion"
let heartbeatsEnabledKey = "marv.heartbeatsEnabled"
let debugPaneEnabledKey = "marv.debugPaneEnabled"
let debugFileLogEnabledKey = "marv.debug.fileLogEnabled"
let appLogLevelKey = "marv.debug.appLogLevel"
let voiceWakeSupported: Bool = ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= 26

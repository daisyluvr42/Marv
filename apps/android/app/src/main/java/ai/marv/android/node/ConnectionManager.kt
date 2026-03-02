package ai.marv.android.node

import android.os.Build
import ai.marv.android.BuildConfig
import ai.marv.android.SecurePrefs
import ai.marv.android.gateway.GatewayClientInfo
import ai.marv.android.gateway.GatewayConnectOptions
import ai.marv.android.gateway.GatewayEndpoint
import ai.marv.android.gateway.GatewayTlsParams
import ai.marv.android.protocol.MarvCanvasA2UICommand
import ai.marv.android.protocol.MarvCanvasCommand
import ai.marv.android.protocol.MarvCameraCommand
import ai.marv.android.protocol.MarvLocationCommand
import ai.marv.android.protocol.MarvScreenCommand
import ai.marv.android.protocol.MarvSmsCommand
import ai.marv.android.protocol.MarvCapability
import ai.marv.android.LocationMode
import ai.marv.android.VoiceWakeMode

class ConnectionManager(
  private val prefs: SecurePrefs,
  private val cameraEnabled: () -> Boolean,
  private val locationMode: () -> LocationMode,
  private val voiceWakeMode: () -> VoiceWakeMode,
  private val smsAvailable: () -> Boolean,
  private val hasRecordAudioPermission: () -> Boolean,
  private val manualTls: () -> Boolean,
) {
  companion object {
    internal fun resolveTlsParamsForEndpoint(
      endpoint: GatewayEndpoint,
      storedFingerprint: String?,
      manualTlsEnabled: Boolean,
    ): GatewayTlsParams? {
      val stableId = endpoint.stableId
      val stored = storedFingerprint?.trim().takeIf { !it.isNullOrEmpty() }
      val isManual = stableId.startsWith("manual|")

      if (isManual) {
        if (!manualTlsEnabled) return null
        if (!stored.isNullOrBlank()) {
          return GatewayTlsParams(
            required = true,
            expectedFingerprint = stored,
            allowTOFU = false,
            stableId = stableId,
          )
        }
        return GatewayTlsParams(
          required = true,
          expectedFingerprint = null,
          allowTOFU = false,
          stableId = stableId,
        )
      }

      // Prefer stored pins. Never let discovery-provided TXT override a stored fingerprint.
      if (!stored.isNullOrBlank()) {
        return GatewayTlsParams(
          required = true,
          expectedFingerprint = stored,
          allowTOFU = false,
          stableId = stableId,
        )
      }

      val hinted = endpoint.tlsEnabled || !endpoint.tlsFingerprintSha256.isNullOrBlank()
      if (hinted) {
        // TXT is unauthenticated. Do not treat the advertised fingerprint as authoritative.
        return GatewayTlsParams(
          required = true,
          expectedFingerprint = null,
          allowTOFU = false,
          stableId = stableId,
        )
      }

      return null
    }
  }

  fun buildInvokeCommands(): List<String> =
    buildList {
      add(MarvCanvasCommand.Present.rawValue)
      add(MarvCanvasCommand.Hide.rawValue)
      add(MarvCanvasCommand.Navigate.rawValue)
      add(MarvCanvasCommand.Eval.rawValue)
      add(MarvCanvasCommand.Snapshot.rawValue)
      add(MarvCanvasA2UICommand.Push.rawValue)
      add(MarvCanvasA2UICommand.PushJSONL.rawValue)
      add(MarvCanvasA2UICommand.Reset.rawValue)
      add(MarvScreenCommand.Record.rawValue)
      if (cameraEnabled()) {
        add(MarvCameraCommand.Snap.rawValue)
        add(MarvCameraCommand.Clip.rawValue)
      }
      if (locationMode() != LocationMode.Off) {
        add(MarvLocationCommand.Get.rawValue)
      }
      if (smsAvailable()) {
        add(MarvSmsCommand.Send.rawValue)
      }
      if (BuildConfig.DEBUG) {
        add("debug.logs")
        add("debug.ed25519")
      }
      add("app.update")
    }

  fun buildCapabilities(): List<String> =
    buildList {
      add(MarvCapability.Canvas.rawValue)
      add(MarvCapability.Screen.rawValue)
      if (cameraEnabled()) add(MarvCapability.Camera.rawValue)
      if (smsAvailable()) add(MarvCapability.Sms.rawValue)
      if (voiceWakeMode() != VoiceWakeMode.Off && hasRecordAudioPermission()) {
        add(MarvCapability.VoiceWake.rawValue)
      }
      if (locationMode() != LocationMode.Off) {
        add(MarvCapability.Location.rawValue)
      }
    }

  fun resolvedVersionName(): String {
    val versionName = BuildConfig.VERSION_NAME.trim().ifEmpty { "dev" }
    return if (BuildConfig.DEBUG && !versionName.contains("dev", ignoreCase = true)) {
      "$versionName-dev"
    } else {
      versionName
    }
  }

  fun resolveModelIdentifier(): String? {
    return listOfNotNull(Build.MANUFACTURER, Build.MODEL)
      .joinToString(" ")
      .trim()
      .ifEmpty { null }
  }

  fun buildUserAgent(): String {
    val version = resolvedVersionName()
    val release = Build.VERSION.RELEASE?.trim().orEmpty()
    val releaseLabel = if (release.isEmpty()) "unknown" else release
    return "MarvAndroid/$version (Android $releaseLabel; SDK ${Build.VERSION.SDK_INT})"
  }

  fun buildClientInfo(clientId: String, clientMode: String): GatewayClientInfo {
    return GatewayClientInfo(
      id = clientId,
      displayName = prefs.displayName.value,
      version = resolvedVersionName(),
      platform = "android",
      mode = clientMode,
      instanceId = prefs.instanceId.value,
      deviceFamily = "Android",
      modelIdentifier = resolveModelIdentifier(),
    )
  }

  fun buildNodeConnectOptions(): GatewayConnectOptions {
    return GatewayConnectOptions(
      role = "node",
      scopes = emptyList(),
      caps = buildCapabilities(),
      commands = buildInvokeCommands(),
      permissions = emptyMap(),
      client = buildClientInfo(clientId = "marv-android", clientMode = "node"),
      userAgent = buildUserAgent(),
    )
  }

  fun buildOperatorConnectOptions(): GatewayConnectOptions {
    return GatewayConnectOptions(
      role = "operator",
      scopes = listOf("operator.read", "operator.write", "operator.talk.secrets"),
      caps = emptyList(),
      commands = emptyList(),
      permissions = emptyMap(),
      client = buildClientInfo(clientId = "marv-control-ui", clientMode = "ui"),
      userAgent = buildUserAgent(),
    )
  }

  fun resolveTlsParams(endpoint: GatewayEndpoint): GatewayTlsParams? {
    val stored = prefs.loadGatewayTlsFingerprint(endpoint.stableId)
    return resolveTlsParamsForEndpoint(endpoint, storedFingerprint = stored, manualTlsEnabled = manualTls())
  }
}

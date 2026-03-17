#!/usr/bin/env bash
# Build, install, and launch the MarvCompanion iOS app on a connected iPhone.
# Outputs JSON for machine consumption (agent tool integration).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IOS_DIR="$REPO_ROOT/apps/ios"
DERIVED_DATA="$IOS_DIR/.build/derivedData"
CONFIGURATION="Debug"
TARGET_DEVICE=""
LIST_DEVICES=0
SKIP_BUILD=0
DRY_RUN=0

# -- Helpers ------------------------------------------------------------------

die_json() {
  printf '{"status":"error","error":"%s"}\n' "$1" >&2
  exit 1
}

log() {
  printf '%s\n' "$*" >&2
}

have() {
  command -v "$1" >/dev/null 2>&1
}

ensure_xcode_developer() {
  # devicectl and xcodebuild require full Xcode, not just Command Line Tools.
  local dev_path
  dev_path="$(xcode-select -p 2>/dev/null || true)"
  if [[ "$dev_path" == */CommandLineTools* || -z "$dev_path" ]]; then
    # Try to find Xcode.app automatically.
    if [[ -d "/Applications/Xcode.app/Contents/Developer" ]]; then
      # Use DEVELOPER_DIR env var instead of sudo xcode-select -s.
      export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
      log "Using Xcode at $DEVELOPER_DIR"
    else
      die_json "Full Xcode is required (not just Command Line Tools). Install Xcode from the App Store."
    fi
  fi
}

ensure_xcodegen() {
  if ! have xcodegen; then
    if have brew; then
      log "Installing xcodegen via Homebrew..."
      brew install xcodegen
    else
      die_json "xcodegen is required. Install it with: brew install xcodegen"
    fi
  fi
}

# -- Device discovery ---------------------------------------------------------

list_connected_devices() {
  local json_out="/tmp/marv-ios-devices-$$.json"
  xcrun devicectl list devices --json-output "$json_out" >/dev/null 2>&1 || {
    rm -f "$json_out"
    die_json "Failed to list devices via devicectl"
  }

  # Extract connected iPhones/iPads (connectionProperties.transportType = wired or wifi,
  # deviceProperties.deviceType contains iPhone/iPad).
  # jq filters: connected devices with a known UDID.
  if ! have jq; then
    # Fallback: output raw JSON for the agent to parse.
    cat "$json_out"
    rm -f "$json_out"
    return 0
  fi

  jq -r '
    [.result.devices[] |
      select(.connectionProperties.tunnelState == "connected" or
             .connectionProperties.transportType != null) |
      {
        name: .deviceProperties.name,
        udid: .identifier,
        model: .deviceProperties.marketingName,
        osVersion: .deviceProperties.osVersionNumber,
        transport: .connectionProperties.transportType
      }
    ]' "$json_out" 2>/dev/null || cat "$json_out"

  rm -f "$json_out"
}

resolve_device_udid() {
  local json_out="/tmp/marv-ios-devices-$$.json"
  xcrun devicectl list devices --json-output "$json_out" >/dev/null 2>&1 || {
    rm -f "$json_out"
    die_json "Failed to list devices via devicectl"
  }

  local udid=""
  local device_name=""

  if have jq; then
    if [[ -n "$TARGET_DEVICE" ]]; then
      # Match by name (substring, case-insensitive) or exact UDID.
      udid=$(jq -r --arg target "$TARGET_DEVICE" '
        [.result.devices[] |
          select(
            (.connectionProperties.tunnelState == "connected" or
             .connectionProperties.transportType != null) and
            ((.deviceProperties.name | ascii_downcase | contains($target | ascii_downcase)) or
             .identifier == $target)
          )] | first | .identifier // empty
      ' "$json_out" 2>/dev/null || true)
      device_name=$(jq -r --arg target "$TARGET_DEVICE" '
        [.result.devices[] |
          select(
            (.connectionProperties.tunnelState == "connected" or
             .connectionProperties.transportType != null) and
            ((.deviceProperties.name | ascii_downcase | contains($target | ascii_downcase)) or
             .identifier == $target)
          )] | first | .deviceProperties.name // empty
      ' "$json_out" 2>/dev/null || true)
    else
      # Pick first connected device.
      udid=$(jq -r '
        [.result.devices[] |
          select(.connectionProperties.tunnelState == "connected" or
                 .connectionProperties.transportType != null)
        ] | first | .identifier // empty
      ' "$json_out" 2>/dev/null || true)
      device_name=$(jq -r '
        [.result.devices[] |
          select(.connectionProperties.tunnelState == "connected" or
                 .connectionProperties.transportType != null)
        ] | first | .deviceProperties.name // empty
      ' "$json_out" 2>/dev/null || true)
    fi
  fi

  rm -f "$json_out"

  if [[ -z "$udid" ]]; then
    if [[ -n "$TARGET_DEVICE" ]]; then
      die_json "No connected device matching '$TARGET_DEVICE'. Run with --list-devices to see available devices."
    else
      die_json "No connected iOS device found. Connect an iPhone via USB and try again."
    fi
  fi

  RESOLVED_UDID="$udid"
  RESOLVED_DEVICE_NAME="$device_name"
}

# -- Build & deploy -----------------------------------------------------------

do_build() {
  log "Configuring signing..."
  bash "$SCRIPT_DIR/ios-configure-signing.sh"

  log "Generating Xcode project..."
  (cd "$IOS_DIR" && xcodegen generate --quiet 2>/dev/null || xcodegen generate)

  log "Building MarvCompanion ($CONFIGURATION) for device $RESOLVED_DEVICE_NAME ($RESOLVED_UDID)..."
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] xcodebuild build -scheme MarvCompanion -configuration $CONFIGURATION -destination id=$RESOLVED_UDID -derivedDataPath $DERIVED_DATA -quiet"
    return 0
  fi

  xcodebuild build \
    -scheme MarvCompanion \
    -configuration "$CONFIGURATION" \
    -destination "id=$RESOLVED_UDID" \
    -derivedDataPath "$DERIVED_DATA" \
    -quiet 2>&1 | tail -5 >&2 || {
    die_json "xcodebuild failed. Check Xcode signing and device trust settings."
  }
}

locate_app_bundle() {
  local app_path
  app_path="$(find "$DERIVED_DATA" -name "MarvCompanion.app" -type d -path "*/Debug-iphoneos/*" 2>/dev/null | head -1)"
  if [[ -z "$app_path" ]]; then
    app_path="$(find "$DERIVED_DATA" -name "MarvCompanion.app" -type d 2>/dev/null | head -1)"
  fi
  if [[ -z "$app_path" ]]; then
    die_json "Could not find MarvCompanion.app in build output"
  fi
  printf '%s' "$app_path"
}

do_install() {
  local app_path="$1"

  log "Installing to $RESOLVED_DEVICE_NAME..."
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] xcrun devicectl device install app --device $RESOLVED_UDID $app_path"
    return 0
  fi

  xcrun devicectl device install app \
    --device "$RESOLVED_UDID" \
    "$app_path" 2>&1 | tail -3 >&2 || {
    die_json "Failed to install app on device. Ensure the device is unlocked and trusts this computer."
  }
}

do_launch() {
  local bundle_id
  bundle_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$1/Info.plist" 2>/dev/null || echo "ai.marv.ios")"

  log "Launching $bundle_id on $RESOLVED_DEVICE_NAME..."
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[dry-run] xcrun devicectl device process launch --device $RESOLVED_UDID $bundle_id"
    return 0
  fi

  xcrun devicectl device process launch \
    --device "$RESOLVED_UDID" \
    "$bundle_id" 2>&1 | tail -3 >&2 || {
    # Launch failure is non-fatal; the app may still have been installed.
    log "warning: Failed to auto-launch the app. Open it manually on the device." >&2
  }

  printf '{"status":"ok","device":"%s","udid":"%s","bundleId":"%s"}\n' \
    "$RESOLVED_DEVICE_NAME" "$RESOLVED_UDID" "$bundle_id"
}

# -- Argument parsing ---------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      TARGET_DEVICE="${2:-}"
      shift 2
      ;;
    --configuration)
      CONFIGURATION="${2:-Debug}"
      shift 2
      ;;
    --list-devices)
      LIST_DEVICES=1
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      cat >&2 <<'HELP'
Usage: ios-deploy.sh [options]

Build and deploy MarvCompanion to a connected iPhone.

Options:
  --device <name|udid>   Target device (default: first connected iPhone)
  --configuration <cfg>  Debug (default) or Release
  --list-devices         List connected iOS devices and exit (JSON)
  --skip-build           Skip build; install previously built .app
  --dry-run              Print commands without executing
  -h, --help             Show this help
HELP
      exit 0
      ;;
    *)
      die_json "Unknown argument: $1"
      ;;
  esac
done

# -- Main ---------------------------------------------------------------------

ensure_xcode_developer
ensure_xcodegen

if [[ "$LIST_DEVICES" == "1" ]]; then
  list_connected_devices
  exit 0
fi

resolve_device_udid

if [[ "$SKIP_BUILD" == "0" ]]; then
  do_build
fi

APP_PATH="$(locate_app_bundle)"
do_install "$APP_PATH"
do_launch "$APP_PATH"

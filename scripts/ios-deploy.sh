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
AUTO_CONFIRM=0
FORCE_DEPLOY=0
DEPLOY_MARKER_DIR="$IOS_DIR/.build"

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

  if ! have jq; then
    rm -f "$json_out"
    die_json "jq is required for device selection. Install it with: brew install jq"
  fi

  # Collect all connected devices into parallel arrays.
  local -a device_udids device_names device_models device_versions
  local count=0

  while IFS=$'\t' read -r d_udid d_name d_model d_version; do
    device_udids+=("$d_udid")
    device_names+=("$d_name")
    device_models+=("$d_model")
    device_versions+=("$d_version")
    count=$((count + 1))
  done < <(jq -r '
    [.result.devices[] |
      select(.connectionProperties.tunnelState == "connected" or
             .connectionProperties.transportType != null)] |
    .[] | [.identifier, .deviceProperties.name,
           .deviceProperties.marketingName, .deviceProperties.osVersionNumber] |
    @tsv
  ' "$json_out" 2>/dev/null || true)

  rm -f "$json_out"

  # Filter by --device flag if provided.
  if [[ -n "$TARGET_DEVICE" ]]; then
    local -a filtered_udids filtered_names filtered_models filtered_versions
    local target_lower
    target_lower="$(printf '%s' "$TARGET_DEVICE" | tr '[:upper:]' '[:lower:]')"
    for i in $(seq 0 $((count - 1))); do
      local name_lower
      name_lower="$(printf '%s' "${device_names[$i]}" | tr '[:upper:]' '[:lower:]')"
      if [[ "${device_udids[$i]}" == "$TARGET_DEVICE" ]] || [[ "$name_lower" == *"$target_lower"* ]]; then
        filtered_udids+=("${device_udids[$i]}")
        filtered_names+=("${device_names[$i]}")
        filtered_models+=("${device_models[$i]}")
        filtered_versions+=("${device_versions[$i]}")
      fi
    done
    device_udids=("${filtered_udids[@]+"${filtered_udids[@]}"}")
    device_names=("${filtered_names[@]+"${filtered_names[@]}"}")
    device_models=("${filtered_models[@]+"${filtered_models[@]}"}")
    device_versions=("${filtered_versions[@]+"${filtered_versions[@]}"}")
    count=${#device_udids[@]}
  fi

  if [[ "$count" -eq 0 ]]; then
    if [[ -n "$TARGET_DEVICE" ]]; then
      die_json "No connected device matching '$TARGET_DEVICE'. Run with --list-devices to see available devices."
    else
      die_json "No connected iOS device found. Connect an iPhone via USB and try again."
    fi
  fi

  local chosen=0

  if [[ "$count" -eq 1 ]]; then
    # Single device: confirm with user (unless --yes).
    log ""
    log "Found device:"
    log "  ${device_names[0]} (${device_models[0]}, iOS ${device_versions[0]})"
    log ""
    if [[ "$AUTO_CONFIRM" != "1" ]]; then
      printf 'Deploy to this device? [Y/n] ' >&2
      local reply
      read -r reply </dev/tty
      case "$reply" in
        [nN]*)
          die_json "Cancelled by user."
          ;;
      esac
    fi
  else
    # Multiple devices: let user choose (unless --yes picks first).
    if [[ "$AUTO_CONFIRM" == "1" ]]; then
      log "Multiple devices found; --yes specified, using first device: ${device_names[0]}"
    else
      log ""
      log "Multiple devices found:"
      for i in $(seq 0 $((count - 1))); do
        printf '  [%d] %s (%s, iOS %s)\n' "$((i + 1))" "${device_names[$i]}" "${device_models[$i]}" "${device_versions[$i]}" >&2
      done
      log ""
      printf 'Select device [1-%d]: ' "$count" >&2
      local selection
      read -r selection </dev/tty
      if [[ ! "$selection" =~ ^[0-9]+$ ]] || [[ "$selection" -lt 1 ]] || [[ "$selection" -gt "$count" ]]; then
        die_json "Invalid selection: $selection"
      fi
      chosen=$((selection - 1))
    fi
  fi

  RESOLVED_UDID="${device_udids[$chosen]}"
  RESOLVED_DEVICE_NAME="${device_names[$chosen]}"
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
    -y|--yes)
      AUTO_CONFIRM=1
      shift
      ;;
    -f|--force)
      FORCE_DEPLOY=1
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
  -y, --yes              Skip device confirmation prompt
  -f, --force            Deploy even if already up-to-date
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

# Skip deploy if the same commit was already deployed to this device.
CURRENT_COMMIT="$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || echo "unknown")"
MARKER_FILE="$DEPLOY_MARKER_DIR/.deployed-${RESOLVED_UDID}"

if [[ "$FORCE_DEPLOY" != "1" && -f "$MARKER_FILE" ]]; then
  last_commit="$(cat "$MARKER_FILE" 2>/dev/null || true)"
  if [[ "$last_commit" == "$CURRENT_COMMIT" ]]; then
    log "Already deployed (commit ${CURRENT_COMMIT:0:7}) to $RESOLVED_DEVICE_NAME. Use --force to redeploy."
    printf '{"status":"ok","device":"%s","udid":"%s","skipped":"already_deployed"}\n' \
      "$RESOLVED_DEVICE_NAME" "$RESOLVED_UDID"
    exit 0
  fi
fi

if [[ "$SKIP_BUILD" == "0" ]]; then
  do_build
fi

APP_PATH="$(locate_app_bundle)"
do_install "$APP_PATH"
do_launch "$APP_PATH"

# Record successful deploy.
mkdir -p "$DEPLOY_MARKER_DIR"
printf '%s' "$CURRENT_COMMIT" > "$MARKER_FILE"

#!/usr/bin/env bash
set -euo pipefail

PACKAGE_NAME="${MARV_PACKAGE_NAME:-agentmarv}"
INSTALL_METHOD="${MARV_INSTALL_METHOD:-npm}"
VERSION="${MARV_VERSION:-latest}"
PACKAGE_SPEC="${MARV_PACKAGE:-}"
REPO_DEFAULT="https://github.com/daisyluvr42/Marv.git"
REPO="${MARV_REPO:-$REPO_DEFAULT}"
REF="${MARV_REF:-main}"
RUN_ONBOARD=1
NO_GIT_UPDATE=0
DRY_RUN=0
VERBOSE=0
FORCE_HOME_NPM_PREFIX=0
# On macOS, default to downloading and launching the Mac app for GUI onboarding.
if [[ "$OSTYPE" == darwin* ]]; then
  INSTALL_MAC_APP="${MARV_NO_MAC_APP:+0}"
  INSTALL_MAC_APP="${INSTALL_MAC_APP:-1}"
else
  INSTALL_MAC_APP=0
fi
SHARP_IGNORE_GLOBAL_LIBVIPS="${SHARP_IGNORE_GLOBAL_LIBVIPS:-1}"

log() {
  printf '%s\n' "$*"
}

warn() {
  printf 'warning: %s\n' "$*" >&2
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Marv installer (macOS/Linux)

Usage:
  install.sh [options]

Options:
  --install-method <npm|git>  Install from npm release (default) or git checkout
  --version <tag|version>     npm dist-tag or version (default: latest)
  --package <path-or-url>     Install from a local/remote .tgz instead of npm registry
  --repo <url>                Git repository URL for --install-method git
  --ref <ref>                 Git ref/branch/tag for --install-method git
  --no-git-update             Reserved for compatibility; ignored by npm installs
  --set-npm-prefix            Force npm global prefix to ~/.npm-global when needed
  --no-onboard                Skip `marv onboard --install-daemon`
  --onboard                   Run onboarding after install (default)
  --no-mac-app                Skip Mac app download; use CLI onboarding on macOS
  --beta                      Shortcut for --version beta
  --dry-run                   Print actions without executing them
  --verbose                   Enable shell tracing
  -h, --help                  Show this help

Examples:
  curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash
  curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash -s -- --beta
  curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash -s -- --install-method git
  curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash -s -- --package ./agentmarv-2026.3.15.tgz

Env:
  MARV_INSTALL_METHOD         npm or git
  MARV_VERSION                npm version or dist-tag
  MARV_PACKAGE                Local/remote package path or URL
  MARV_REPO                   Git repository URL
  MARV_REF                    Git ref
  MARV_NO_ONBOARD=1           Skip onboarding
  MARV_NO_MAC_APP=1           Skip Mac app download on macOS
  MARV_DRY_RUN=1              Print commands only
  MARV_VERBOSE=1              Enable shell tracing
USAGE
}

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] %s\n' "$*"
    return 0
  fi
  "$@"
}

run_bash_snippet() {
  local snippet="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] bash -lc %q\n' "$snippet"
    return 0
  fi
  bash -lc "$snippet"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

sudo_run() {
  if [[ "$(id -u)" == "0" ]]; then
    run "$@"
    return 0
  fi
  if ! have sudo; then
    die "sudo is required to install system dependencies automatically."
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] sudo %s\n' "$*"
    return 0
  fi
  sudo "$@"
}

append_line_if_missing() {
  local file="$1"
  local line="$2"
  # Create the rc file if it doesn't exist yet (e.g. fresh OS install).
  if [[ ! -f "$file" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '[dry-run] touch %s\n' "$file"
    else
      touch "$file"
    fi
  fi
  grep -Fqx "$line" "$file" 2>/dev/null && return 0
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] append %q to %s\n' "$line" "$file"
    return 0
  fi
  printf '\n%s\n' "$line" >>"$file"
}

ensure_home_npm_prefix() {
  local target_prefix="$HOME/.npm-global"
  local current_prefix writable=0
  current_prefix="$(npm config get prefix 2>/dev/null || true)"

  if [[ -n "$current_prefix" && -d "$current_prefix" && -w "$current_prefix" ]]; then
    writable=1
  fi
  if [[ -n "$current_prefix" && -d "$current_prefix/lib/node_modules" && -w "$current_prefix/lib/node_modules" ]]; then
    writable=1
  fi

  if [[ "$FORCE_HOME_NPM_PREFIX" != "1" && "$writable" == "1" ]]; then
    return 0
  fi

  log "Configuring npm global prefix at $target_prefix"
  run mkdir -p "$target_prefix/bin"
  run npm config set prefix "$target_prefix"
  export PATH="$target_prefix/bin:$PATH"
  append_line_if_missing "$HOME/.bashrc" 'export PATH="$HOME/.npm-global/bin:$PATH"'
  append_line_if_missing "$HOME/.zshrc" 'export PATH="$HOME/.npm-global/bin:$PATH"'
  append_line_if_missing "$HOME/.profile" 'export PATH="$HOME/.npm-global/bin:$PATH"'
}

ensure_git() {
  if have git; then
    return 0
  fi
  log "Git not found; installing git"
  if have apt-get; then
    sudo_run apt-get update
    sudo_run apt-get install -y git
    return 0
  fi
  if have dnf; then
    sudo_run dnf install -y git
    return 0
  fi
  if have yum; then
    sudo_run yum install -y git
    return 0
  fi
  if [[ "$OSTYPE" == darwin* ]]; then
    if ! have brew; then
      die 'Git is missing and Homebrew is not installed. Install Homebrew first, then rerun the installer.'
    fi
    run brew install git
    return 0
  fi
  die "Git is required but could not be installed automatically on this system."
}

node_major() {
  local version major
  version="$(node -v 2>/dev/null || true)"
  major="$(printf '%s' "$version" | sed -E 's/^v([0-9]+).*/\1/')"
  if [[ "$major" =~ ^[0-9]+$ ]]; then
    printf '%s' "$major"
    return 0
  fi
  printf '0'
}

ensure_node_22() {
  if have node && [[ "$(node_major)" -ge 22 ]]; then
    return 0
  fi

  log "Node.js 22+ not found; installing Node.js 22"
  if [[ "$OSTYPE" == darwin* ]]; then
    if ! have brew; then
      die 'Node.js 22+ is missing and Homebrew is not installed. Install Homebrew first, then rerun the installer.'
    fi
    run brew install node@22
    export PATH="$(brew --prefix node@22)/bin:$PATH"
  elif have apt-get; then
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '[dry-run] curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -\n'
    else
      curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    fi
    sudo_run apt-get install -y nodejs
  elif have dnf; then
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '[dry-run] curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo bash -\n'
    else
      curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo bash -
    fi
    sudo_run dnf install -y nodejs
  elif have yum; then
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '[dry-run] curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo bash -\n'
    else
      curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo bash -
    fi
    sudo_run yum install -y nodejs
  else
    die "Node.js 22+ is required and this installer does not know how to install it on this system."
  fi

  have node || die "Node.js installation finished, but node is still not on PATH."
  have npm || die "npm is missing after Node.js installation."
  if [[ "$(node_major)" -lt 22 ]]; then
    die "Node.js 22+ is required. Current version: $(node -v 2>/dev/null || echo unknown)"
  fi
}

install_mac_app() {
  local marv_ver app_plist installed_ver zip_url zip_path
  marv_ver="$("$CLI_PATH" --version 2>/dev/null | head -1 || true)"
  if [[ -z "$marv_ver" ]]; then
    warn "Could not determine Marv version for Mac app download."
    return 1
  fi

  # Check if already installed with matching version.
  app_plist="/Applications/Marv.app/Contents/Info.plist"
  if [[ -f "$app_plist" ]]; then
    installed_ver="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app_plist" 2>/dev/null || true)"
    if [[ "$installed_ver" == "$marv_ver" ]]; then
      log "Marv.app $marv_ver already installed."
      run open /Applications/Marv.app
      return 0
    fi
    log "Updating Marv.app from $installed_ver to $marv_ver"
  fi

  zip_url="https://github.com/daisyluvr42/Marv/releases/download/v${marv_ver}/Marv-${marv_ver}.zip"
  sha_url="https://github.com/daisyluvr42/Marv/releases/download/v${marv_ver}/Marv-${marv_ver}.zip.sha256"
  zip_path="${TMPDIR:-/tmp}/Marv-${marv_ver}.zip"

  log "Downloading Marv.app $marv_ver..."
  local attempt max_attempts=3
  for attempt in $(seq 1 $max_attempts); do
    if curl -fsSL --retry 2 -o "$zip_path" "$zip_url"; then
      break
    fi
    if [[ "$attempt" -lt "$max_attempts" ]]; then
      warn "Download attempt $attempt/$max_attempts failed; retrying..."
      sleep 1
    else
      warn "Failed to download Mac app from $zip_url after $max_attempts attempts"
      rm -f "$zip_path"
      return 1
    fi
  done

  # Verify SHA-256 checksum if available.
  local expected_sha
  expected_sha="$(curl -fsSL "$sha_url" 2>/dev/null | awk '{print $1}' || true)"
  if [[ -n "$expected_sha" ]]; then
    local actual_sha
    actual_sha="$(shasum -a 256 "$zip_path" | awk '{print $1}')"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
      warn "Checksum mismatch for Marv.app download (expected $expected_sha, got $actual_sha)"
      rm -f "$zip_path"
      return 1
    fi
    log "Checksum verified."
  fi

  log "Installing Marv.app to /Applications..."
  run ditto -xk "$zip_path" /Applications/
  rm -f "$zip_path"
  # Remove quarantine so the app launches without Gatekeeper prompt.
  xattr -dr com.apple.quarantine /Applications/Marv.app 2>/dev/null || true

  log "Launching Marv.app..."
  run open /Applications/Marv.app
  return 0
}

resolve_install_spec() {
  if [[ -n "$PACKAGE_SPEC" ]]; then
    printf '%s' "$PACKAGE_SPEC"
    return 0
  fi

  case "$INSTALL_METHOD" in
    npm)
      printf '%s@%s' "$PACKAGE_NAME" "$VERSION"
      ;;
    git)
      if [[ "$REPO" == git+* ]]; then
        printf '%s#%s' "$REPO" "$REF"
      else
        printf 'git+%s#%s' "$REPO" "$REF"
      fi
      ;;
    *)
      die "Unsupported --install-method: $INSTALL_METHOD"
      ;;
  esac
}

resolve_cli_path() {
  if have marv; then
    command -v marv
    return 0
  fi
  local npm_prefix
  npm_prefix="$(npm config get prefix 2>/dev/null || true)"
  if [[ -n "$npm_prefix" && -x "$npm_prefix/bin/marv" ]]; then
    printf '%s' "$npm_prefix/bin/marv"
    return 0
  fi
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-method|--method)
      INSTALL_METHOD="${2:-}"
      shift 2
      ;;
    --version)
      VERSION="${2:-}"
      shift 2
      ;;
    --beta)
      VERSION="beta"
      shift
      ;;
    --package)
      PACKAGE_SPEC="${2:-}"
      shift 2
      ;;
    --repo)
      REPO="${2:-}"
      shift 2
      ;;
    --ref)
      REF="${2:-}"
      shift 2
      ;;
    --no-git-update)
      NO_GIT_UPDATE=1
      shift
      ;;
    --set-npm-prefix)
      FORCE_HOME_NPM_PREFIX=1
      shift
      ;;
    --no-onboard)
      RUN_ONBOARD=0
      shift
      ;;
    --onboard)
      RUN_ONBOARD=1
      shift
      ;;
    --no-mac-app)
      INSTALL_MAC_APP=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

if [[ "${MARV_NO_ONBOARD:-0}" == "1" ]]; then
  RUN_ONBOARD=0
fi
if [[ "${MARV_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1
fi
if [[ "${MARV_VERBOSE:-0}" == "1" ]]; then
  VERBOSE=1
fi

if [[ "$VERBOSE" == "1" ]]; then
  set -x
fi

ensure_node_22
ensure_git
ensure_home_npm_prefix

if [[ "$NO_GIT_UPDATE" == "1" && "$INSTALL_METHOD" != "git" ]]; then
  warn "--no-git-update only applies to --install-method git"
fi

INSTALL_SPEC="$(resolve_install_spec)"
log "Installing Marv from: $INSTALL_SPEC"
export SHARP_IGNORE_GLOBAL_LIBVIPS
export NPM_CONFIG_FUND=false
export NPM_CONFIG_AUDIT=false

run npm install -g "$INSTALL_SPEC"

if [[ "$DRY_RUN" == "1" ]]; then
  if [[ "$INSTALL_MAC_APP" == "1" ]]; then
    log "[dry-run] Would download and install Marv.app from GitHub Release"
    log "[dry-run] Would launch /Applications/Marv.app for GUI onboarding"
  elif [[ "$RUN_ONBOARD" == "1" ]]; then
    log "[dry-run] Would run: marv onboard --install-daemon"
  fi
  log "Dry run complete."
  exit 0
fi

CLI_PATH="$(resolve_cli_path || true)"
if [[ -z "$CLI_PATH" ]]; then
  die "Install finished, but the marv CLI was not found on PATH. Try opening a new terminal and run: marv --version"
fi

log "CLI ready: $CLI_PATH"
run "$CLI_PATH" --version

if [[ "$RUN_ONBOARD" == "1" ]]; then
  if [[ "$INSTALL_MAC_APP" == "1" ]]; then
    if [[ "$INSTALL_METHOD" == "git" ]]; then
      # Git install: source is available locally — build Mac app from source
      # instead of downloading a pre-built (unsigned) binary.
      log ""
      log "────────────────────────────────────────────────────"
      log "  Marv CLI installed successfully."
      log ""
      log "  To install the Mac menu-bar app, run:"
      log "    scripts/setup-mac-app.sh"
      log ""
      log "  (Requires Xcode Command Line Tools)"
      log "────────────────────────────────────────────────────"
      log ""
      run "$CLI_PATH" onboard --install-daemon
    elif install_mac_app; then
      log "Marv.app installed and launched. Complete setup in the app."
    else
      warn "Mac app download failed; falling back to CLI onboarding."
      run "$CLI_PATH" onboard --install-daemon
    fi
  else
    log "Running onboarding..."
    run "$CLI_PATH" onboard --install-daemon
  fi
else
  log "Install complete."
  log "Run this when ready:"
  log "  $CLI_PATH onboard --install-daemon"
fi

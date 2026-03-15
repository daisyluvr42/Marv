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
  --beta                      Shortcut for --version beta
  --dry-run                   Print actions without executing them
  --verbose                   Enable shell tracing
  -h, --help                  Show this help

Examples:
  curl -fsSL https://marv.bot/install.sh | bash
  curl -fsSL https://marv.bot/install.sh | bash -s -- --beta
  curl -fsSL https://marv.bot/install.sh | bash -s -- --install-method git
  curl -fsSL https://marv.bot/install.sh | bash -s -- --package ./agentmarv-2026.3.15.tgz

Env:
  MARV_INSTALL_METHOD         npm or git
  MARV_VERSION                npm version or dist-tag
  MARV_PACKAGE                Local/remote package path or URL
  MARV_REPO                   Git repository URL
  MARV_REF                    Git ref
  MARV_NO_ONBOARD=1           Skip onboarding
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
  [[ -f "$file" ]] || return 0
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
  log "Running onboarding..."
  run "$CLI_PATH" onboard --install-daemon
else
  log "Install complete."
  log "Run this when ready:"
  log "  $CLI_PATH onboard --install-daemon"
fi

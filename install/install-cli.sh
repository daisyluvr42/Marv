#!/usr/bin/env bash
set -euo pipefail

PREFIX="${MARV_PREFIX:-$HOME/.marv}"
PACKAGE_NAME="${MARV_PACKAGE_NAME:-agentmarv}"
VERSION="${MARV_VERSION:-latest}"
NODE_VERSION="${MARV_NODE_VERSION:-22.22.0}"
PACKAGE_SPEC="${MARV_PACKAGE:-}"
RUN_ONBOARD=0
JSON_OUTPUT=0
SET_NPM_PREFIX=0

log() {
  printf '%s\n' "$*"
}

json_event() {
  [[ "$JSON_OUTPUT" == "1" ]] || return 0
  local phase detail
  phase="${1//\\/\\\\}"
  phase="${phase//\"/\\\"}"
  detail="${2//\\/\\\\}"
  detail="${detail//\"/\\\"}"
  printf '{"phase":"%s","detail":"%s"}\n' "$phase" "$detail"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Marv local-prefix installer (macOS/Linux)

Usage:
  install-cli.sh [options]

Options:
  --prefix <path>            Install prefix (default: ~/.marv)
  --version <tag|version>    npm dist-tag or version (default: latest)
  --package <path-or-url>    Install from a local/remote .tgz instead of npm registry
  --node-version <version>   Node.js version to download locally (default: 22.22.0)
  --json                     Emit NDJSON progress events
  --onboard                  Run `marv onboard --install-daemon` after install
  --no-onboard               Skip onboarding (default)
  --set-npm-prefix           Accepted for compatibility; local-prefix installs do not need it
  -h, --help                 Show this help

Examples:
  curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install-cli.sh | bash
  curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install-cli.sh | bash -s -- --prefix ~/Documents/Marv-Run
  curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install-cli.sh | bash -s -- --package ./agentmarv-2026.3.15.tgz --prefix ~/Documents/Marv-Run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="${2:-}"
      shift 2
      ;;
    --version)
      VERSION="${2:-}"
      shift 2
      ;;
    --package)
      PACKAGE_SPEC="${2:-}"
      shift 2
      ;;
    --node-version)
      NODE_VERSION="${2:-}"
      shift 2
      ;;
    --json)
      JSON_OUTPUT=1
      shift
      ;;
    --onboard)
      RUN_ONBOARD=1
      shift
      ;;
    --no-onboard)
      RUN_ONBOARD=0
      shift
      ;;
    --set-npm-prefix)
      SET_NPM_PREFIX=1
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

have() {
  command -v "$1" >/dev/null 2>&1
}

resolve_platform() {
  case "$(uname -s)" in
    Darwin) printf 'darwin' ;;
    Linux) printf 'linux' ;;
    *) die "Unsupported OS: $(uname -s)" ;;
  esac
}

resolve_arch() {
  case "$(uname -m)" in
    x86_64|amd64) printf 'x64' ;;
    arm64|aarch64) printf 'arm64' ;;
    *) die "Unsupported CPU architecture: $(uname -m)" ;;
  esac
}

resolve_install_spec() {
  if [[ -n "$PACKAGE_SPEC" ]]; then
    printf '%s' "$PACKAGE_SPEC"
    return 0
  fi
  printf '%s@%s' "$PACKAGE_NAME" "$VERSION"
}

ensure_local_node() {
  local platform arch archive_url cache_dir archive_path tools_root version_dir link_dir extracted_dir
  platform="$(resolve_platform)"
  arch="$(resolve_arch)"
  tools_root="$PREFIX/tools"
  version_dir="$tools_root/node-v$NODE_VERSION"
  link_dir="$tools_root/node"
  if [[ -x "$link_dir/bin/node" ]]; then
    return 0
  fi

  archive_url="https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-$platform-$arch.tar.gz"
  cache_dir="$PREFIX/.cache"
  archive_path="$cache_dir/node-v$NODE_VERSION-$platform-$arch.tar.gz"
  extracted_dir="$cache_dir/node-v$NODE_VERSION-$platform-$arch"

  json_event "node" "download"
  log "Downloading Node.js $NODE_VERSION ($platform-$arch)"
  mkdir -p "$cache_dir" "$tools_root"
  curl -fsSL "$archive_url" -o "$archive_path"

  json_event "node" "extract"
  rm -rf "$extracted_dir" "$version_dir"
  tar -xzf "$archive_path" -C "$cache_dir"
  mv "$extracted_dir" "$version_dir"
  ln -sfn "node-v$NODE_VERSION" "$link_dir"
}

write_local_wrapper() {
  local bin_dir wrapper_path
  bin_dir="$PREFIX/bin"
  wrapper_path="$bin_dir/marv"
  mkdir -p "$bin_dir" "$PREFIX/home"
  cat >"$wrapper_path" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PREFIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MARV_HOME="${MARV_HOME:-$PREFIX_DIR/home}"
exec "$PREFIX_DIR/tools/node/bin/node" "$PREFIX_DIR/node_modules/agentmarv/marv.mjs" "$@"
EOF
  chmod +x "$wrapper_path"
}

ensure_local_package_json() {
  if [[ -f "$PREFIX/package.json" ]]; then
    return 0
  fi
  mkdir -p "$PREFIX"
  cat >"$PREFIX/package.json" <<'EOF'
{
  "name": "marv-local-install",
  "private": true
}
EOF
}

INSTALL_SPEC="$(resolve_install_spec)"
mkdir -p "$(dirname "$PREFIX")"
PREFIX="$(cd "$(dirname "$PREFIX")" && pwd)/$(basename "$PREFIX")"
json_event "prefix" "$PREFIX"
mkdir -p "$PREFIX"

if [[ "$SET_NPM_PREFIX" == "1" ]]; then
  log "Note: --set-npm-prefix is ignored for local-prefix installs; the CLI is isolated under $PREFIX"
fi

ensure_local_node
ensure_local_package_json
write_local_wrapper

json_event "install" "$INSTALL_SPEC"
log "Installing Marv into $PREFIX"
export SHARP_IGNORE_GLOBAL_LIBVIPS="${SHARP_IGNORE_GLOBAL_LIBVIPS:-1}"
export NPM_CONFIG_FUND=false
export NPM_CONFIG_AUDIT=false
"$PREFIX/tools/node/bin/npm" install --prefix "$PREFIX" "$INSTALL_SPEC"

write_local_wrapper
CLI_PATH="$PREFIX/bin/marv"
[[ -x "$CLI_PATH" ]] || die "Local CLI wrapper was not created: $CLI_PATH"

json_event "verify" "$CLI_PATH"
log "CLI ready: $CLI_PATH"
"$CLI_PATH" --version
log "This local install uses: MARV_HOME=$PREFIX/home"
log "Run CLI commands with:"
log "  $CLI_PATH ..."

if [[ "$RUN_ONBOARD" == "1" ]]; then
  json_event "onboard" "$CLI_PATH"
  log "Running onboarding..."
  "$CLI_PATH" onboard --install-daemon
fi

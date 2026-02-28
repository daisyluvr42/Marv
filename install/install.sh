#!/usr/bin/env bash
set -euo pipefail

REPO_DEFAULT="https://github.com/daisyluvr42/Marv.git"
REPO="${MARV_REPO:-$REPO_DEFAULT}"
REF="${MARV_REF:-main}"
RUN_ONBOARD=1

usage() {
  cat <<'USAGE'
Marv installer (macOS/Linux)

Usage:
  install.sh [--no-onboard] [--onboard] [--repo <url>] [--ref <git-ref>]

Options:
  --no-onboard   Skip `marv onboard --install-daemon`
  --onboard      Run onboarding after install (default)
  --repo <url>   Git repository URL (default: https://github.com/daisyluvr42/Marv.git)
  --ref <ref>    Git ref/branch/tag (default: main)
  -h, --help     Show this help

Env:
  MARV_REPO      Override repo URL
  MARV_REF       Override git ref
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-onboard)
      RUN_ONBOARD=0
      shift
      ;;
    --onboard)
      RUN_ONBOARD=1
      shift
      ;;
    --repo)
      REPO="${2:-}"
      shift 2
      ;;
    --ref)
      REF="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required (22+). Please install Node first." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required. Please install npm first." >&2
  exit 1
fi

NODE_VERSION="$(node -v 2>/dev/null || true)"
NODE_MAJOR="$(echo "$NODE_VERSION" | sed -E 's/^v([0-9]+).*/\1/')"
if [[ -z "$NODE_MAJOR" || ! "$NODE_MAJOR" =~ ^[0-9]+$ || "$NODE_MAJOR" -lt 22 ]]; then
  echo "Node.js 22+ is required. Current version: ${NODE_VERSION:-unknown}" >&2
  exit 1
fi

if [[ "$REPO" == git+* ]]; then
  PKG_SPEC="${REPO}#${REF}"
else
  PKG_SPEC="git+${REPO}#${REF}"
fi

echo "Installing Marv from: $PKG_SPEC"
npm install -g "$PKG_SPEC"

if ! command -v marv >/dev/null 2>&1; then
  echo "Install finished, but \`marv\` is not on PATH yet." >&2
  echo "Try reopening your terminal, then run: marv --version" >&2
  exit 0
fi

if [[ "$RUN_ONBOARD" -eq 1 ]]; then
  echo "Running onboarding..."
  marv onboard --install-daemon
else
  echo "Install complete. Run this when ready:"
  echo "  marv onboard --install-daemon"
fi

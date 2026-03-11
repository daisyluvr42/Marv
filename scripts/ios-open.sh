#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IOS_DIR="${ROOT_DIR}/apps/ios"
GENERATE_ONLY=0

if [[ "${1:-}" == "--generate-only" ]]; then
  GENERATE_ONLY=1
fi

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "ERROR: xcodegen is required. Install it with 'brew install xcodegen'." >&2
  exit 1
fi

if command -v xcodebuild >/dev/null 2>&1; then
  if ! xcodebuild -version >/dev/null 2>&1; then
    echo "ERROR: xcodebuild is pointing at CommandLineTools instead of full Xcode." >&2
    echo "Run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
    exit 1
  fi
fi

bash "${ROOT_DIR}/scripts/ios-configure-signing.sh"

cd "${IOS_DIR}"
xcodegen generate

if [[ "${GENERATE_ONLY}" == "1" ]]; then
  exit 0
fi

open MarvCompanion.xcodeproj

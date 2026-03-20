#!/usr/bin/env bash
set -euo pipefail

# Build the Marv Mac app from source and install to /Applications.
# Run this after you have installed the CLI (pnpm install && pnpm build).

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_SRC="$ROOT_DIR/dist/Marv.app"
APP_DEST="/Applications/Marv.app"

log() { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# ── Preflight ──────────────────────────────────────────────────────
if [[ "$OSTYPE" != darwin* ]]; then
  die "This script is macOS-only."
fi

if ! command -v swift >/dev/null 2>&1; then
  die "Xcode Command Line Tools required. Install with: xcode-select --install"
fi

if ! command -v marv >/dev/null 2>&1 && [[ ! -x "$ROOT_DIR/dist/cli.cjs" ]]; then
  warn "Marv CLI not found. Run 'pnpm install && pnpm build' first."
fi

# ── Build ──────────────────────────────────────────────────────────
log "Building Marv.app from source..."
ALLOW_ADHOC_SIGNING=1 "$ROOT_DIR/scripts/package-mac-app.sh"

if [[ ! -d "$APP_SRC" ]]; then
  die "Build failed: $APP_SRC not found."
fi

# ── Install ────────────────────────────────────────────────────────
log "Installing Marv.app to /Applications..."
if [[ -d "$APP_DEST" ]]; then
  log "Removing existing $APP_DEST"
  rm -rf "$APP_DEST"
fi
cp -R "$APP_SRC" "$APP_DEST"

# Remove quarantine so the app launches without Gatekeeper prompt.
xattr -dr com.apple.quarantine "$APP_DEST" 2>/dev/null || true

log "Launching Marv.app..."
open "$APP_DEST"

log ""
log "✅ Marv.app installed and running."
log "   The menu-bar icon should appear shortly."

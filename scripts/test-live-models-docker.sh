#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${MARV_IMAGE:-${CLAWDBOT_IMAGE:-marv:local}}"
CONFIG_DIR="${MARV_CONFIG_DIR:-${CLAWDBOT_CONFIG_DIR:-$HOME/.marv}}"
WORKSPACE_DIR="${MARV_WORKSPACE_DIR:-${CLAWDBOT_WORKSPACE_DIR:-$HOME/.marv/workspace}}"
PROFILE_FILE="${MARV_PROFILE_FILE:-${CLAWDBOT_PROFILE_FILE:-$HOME/.profile}}"

PROFILE_MOUNT=()
if [[ -f "$PROFILE_FILE" ]]; then
  PROFILE_MOUNT=(-v "$PROFILE_FILE":/home/node/.profile:ro)
fi

echo "==> Build image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" -f "$ROOT_DIR/Dockerfile" "$ROOT_DIR"

echo "==> Run live model tests (profile keys)"
docker run --rm -t \
  --entrypoint bash \
  -e COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
  -e HOME=/home/node \
  -e NODE_OPTIONS=--disable-warning=ExperimentalWarning \
  -e MARV_LIVE_TEST=1 \
  -e MARV_LIVE_MODELS="${MARV_LIVE_MODELS:-${CLAWDBOT_LIVE_MODELS:-all}}" \
  -e MARV_LIVE_PROVIDERS="${MARV_LIVE_PROVIDERS:-${CLAWDBOT_LIVE_PROVIDERS:-}}" \
  -e MARV_LIVE_MODEL_TIMEOUT_MS="${MARV_LIVE_MODEL_TIMEOUT_MS:-${CLAWDBOT_LIVE_MODEL_TIMEOUT_MS:-}}" \
  -e MARV_LIVE_REQUIRE_PROFILE_KEYS="${MARV_LIVE_REQUIRE_PROFILE_KEYS:-${CLAWDBOT_LIVE_REQUIRE_PROFILE_KEYS:-}}" \
  -v "$CONFIG_DIR":/home/node/.marv \
  -v "$WORKSPACE_DIR":/home/node/.marv/workspace \
  "${PROFILE_MOUNT[@]}" \
  "$IMAGE_NAME" \
  -lc "set -euo pipefail; [ -f \"$HOME/.profile\" ] && source \"$HOME/.profile\" || true; cd /app && pnpm test:live"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${MARV_INSTALL_E2E_IMAGE:-${CLAWDBOT_INSTALL_E2E_IMAGE:-marv-install-e2e:local}}"
INSTALL_URL="${MARV_INSTALL_URL:-${CLAWDBOT_INSTALL_URL:-https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh}}"

OPENAI_API_KEY="${OPENAI_API_KEY:-}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
ANTHROPIC_API_TOKEN="${ANTHROPIC_API_TOKEN:-}"
MARV_E2E_MODELS="${MARV_E2E_MODELS:-${CLAWDBOT_E2E_MODELS:-}}"

echo "==> Build image: $IMAGE_NAME"
docker build \
  -t "$IMAGE_NAME" \
  -f "$ROOT_DIR/scripts/docker/install-sh-e2e/Dockerfile" \
  "$ROOT_DIR/scripts/docker/install-sh-e2e"

echo "==> Run E2E installer test"
docker run --rm \
  -e MARV_INSTALL_URL="$INSTALL_URL" \
  -e MARV_INSTALL_TAG="${MARV_INSTALL_TAG:-${CLAWDBOT_INSTALL_TAG:-latest}}" \
  -e MARV_E2E_MODELS="$MARV_E2E_MODELS" \
  -e MARV_INSTALL_E2E_PREVIOUS="${MARV_INSTALL_E2E_PREVIOUS:-${CLAWDBOT_INSTALL_E2E_PREVIOUS:-}}" \
  -e MARV_INSTALL_E2E_SKIP_PREVIOUS="${MARV_INSTALL_E2E_SKIP_PREVIOUS:-${CLAWDBOT_INSTALL_E2E_SKIP_PREVIOUS:-0}}" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -e ANTHROPIC_API_TOKEN="$ANTHROPIC_API_TOKEN" \
  "$IMAGE_NAME"

#!/usr/bin/env bash
set -euo pipefail

cd /repo

export MARV_STATE_DIR="/tmp/marv-test"
export MARV_CONFIG_PATH="${MARV_STATE_DIR}/marv.json"

echo "==> Build"
pnpm build

echo "==> Seed state"
mkdir -p "${MARV_STATE_DIR}/credentials"
mkdir -p "${MARV_STATE_DIR}/agents/main/sessions"
echo '{}' >"${MARV_CONFIG_PATH}"
echo 'creds' >"${MARV_STATE_DIR}/credentials/marker.txt"
echo 'session' >"${MARV_STATE_DIR}/agents/main/sessions/sessions.json"

echo "==> Reset (config+creds+sessions)"
pnpm marv reset --scope config+creds+sessions --yes --non-interactive

test ! -f "${MARV_CONFIG_PATH}"
test ! -d "${MARV_STATE_DIR}/credentials"
test ! -d "${MARV_STATE_DIR}/agents/main/sessions"

echo "==> Recreate minimal config"
mkdir -p "${MARV_STATE_DIR}/credentials"
echo '{}' >"${MARV_CONFIG_PATH}"

echo "==> Uninstall (state only)"
pnpm marv uninstall --state --yes --non-interactive

test ! -d "${MARV_STATE_DIR}"

echo "OK"

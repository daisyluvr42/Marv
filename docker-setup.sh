#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker-compose.yml"
EXTRA_COMPOSE_FILE="$ROOT_DIR/docker-compose.extra.yml"
IMAGE_NAME="${MARV_IMAGE:-marv:local}"
EXTRA_MOUNTS="${MARV_EXTRA_MOUNTS:-}"
HOME_VOLUME_NAME="${MARV_HOME_VOLUME:-}"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing dependency: $1" >&2
    exit 1
  fi
}

require_cmd docker
if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose not available (try: docker compose version)" >&2
  exit 1
fi

MARV_CONFIG_DIR="${MARV_CONFIG_DIR:-$HOME/.marv}"
MARV_WORKSPACE_DIR="${MARV_WORKSPACE_DIR:-$HOME/.marv/workspace}"

mkdir -p "$MARV_CONFIG_DIR"
mkdir -p "$MARV_WORKSPACE_DIR"

export MARV_CONFIG_DIR
export MARV_WORKSPACE_DIR
export MARV_GATEWAY_PORT="${MARV_GATEWAY_PORT:-18789}"
export MARV_BRIDGE_PORT="${MARV_BRIDGE_PORT:-18790}"
export MARV_GATEWAY_BIND="${MARV_GATEWAY_BIND:-lan}"
export MARV_IMAGE="$IMAGE_NAME"
export MARV_DOCKER_APT_PACKAGES="${MARV_DOCKER_APT_PACKAGES:-}"
export MARV_EXTRA_MOUNTS="$EXTRA_MOUNTS"
export MARV_HOME_VOLUME="$HOME_VOLUME_NAME"

if [[ -z "${MARV_GATEWAY_TOKEN:-}" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    MARV_GATEWAY_TOKEN="$(openssl rand -hex 32)"
  else
    MARV_GATEWAY_TOKEN="$(python3 - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
)"
  fi
fi
export MARV_GATEWAY_TOKEN

COMPOSE_FILES=("$COMPOSE_FILE")
COMPOSE_ARGS=()

write_extra_compose() {
  local home_volume="$1"
  shift
  local mount

  cat >"$EXTRA_COMPOSE_FILE" <<'YAML'
services:
  marv-gateway:
    volumes:
YAML

  if [[ -n "$home_volume" ]]; then
    printf '      - %s:/home/node\n' "$home_volume" >>"$EXTRA_COMPOSE_FILE"
    printf '      - %s:/home/node/.marv\n' "$MARV_CONFIG_DIR" >>"$EXTRA_COMPOSE_FILE"
    printf '      - %s:/home/node/.marv/workspace\n' "$MARV_WORKSPACE_DIR" >>"$EXTRA_COMPOSE_FILE"
  fi

  for mount in "$@"; do
    printf '      - %s\n' "$mount" >>"$EXTRA_COMPOSE_FILE"
  done

  cat >>"$EXTRA_COMPOSE_FILE" <<'YAML'
  marv-cli:
    volumes:
YAML

  if [[ -n "$home_volume" ]]; then
    printf '      - %s:/home/node\n' "$home_volume" >>"$EXTRA_COMPOSE_FILE"
    printf '      - %s:/home/node/.marv\n' "$MARV_CONFIG_DIR" >>"$EXTRA_COMPOSE_FILE"
    printf '      - %s:/home/node/.marv/workspace\n' "$MARV_WORKSPACE_DIR" >>"$EXTRA_COMPOSE_FILE"
  fi

  for mount in "$@"; do
    printf '      - %s\n' "$mount" >>"$EXTRA_COMPOSE_FILE"
  done

  if [[ -n "$home_volume" && "$home_volume" != *"/"* ]]; then
    cat >>"$EXTRA_COMPOSE_FILE" <<YAML
volumes:
  ${home_volume}:
YAML
  fi
}

VALID_MOUNTS=()
if [[ -n "$EXTRA_MOUNTS" ]]; then
  IFS=',' read -r -a mounts <<<"$EXTRA_MOUNTS"
  for mount in "${mounts[@]}"; do
    mount="${mount#"${mount%%[![:space:]]*}"}"
    mount="${mount%"${mount##*[![:space:]]}"}"
    if [[ -n "$mount" ]]; then
      VALID_MOUNTS+=("$mount")
    fi
  done
fi

if [[ -n "$HOME_VOLUME_NAME" || ${#VALID_MOUNTS[@]} -gt 0 ]]; then
  # Bash 3.2 + nounset treats "${array[@]}" on an empty array as unbound.
  if [[ ${#VALID_MOUNTS[@]} -gt 0 ]]; then
    write_extra_compose "$HOME_VOLUME_NAME" "${VALID_MOUNTS[@]}"
  else
    write_extra_compose "$HOME_VOLUME_NAME"
  fi
  COMPOSE_FILES+=("$EXTRA_COMPOSE_FILE")
fi
for compose_file in "${COMPOSE_FILES[@]}"; do
  COMPOSE_ARGS+=("-f" "$compose_file")
done
COMPOSE_HINT="docker compose"
for compose_file in "${COMPOSE_FILES[@]}"; do
  COMPOSE_HINT+=" -f ${compose_file}"
done

ENV_FILE="$ROOT_DIR/.env"
upsert_env() {
  local file="$1"
  shift
  local -a keys=("$@")
  local tmp
  tmp="$(mktemp)"
  # Use a delimited string instead of an associative array so the script
  # works with Bash 3.2 (macOS default) which lacks `declare -A`.
  local seen=" "

  if [[ -f "$file" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      local key="${line%%=*}"
      local replaced=false
      for k in "${keys[@]}"; do
        if [[ "$key" == "$k" ]]; then
          printf '%s=%s\n' "$k" "${!k-}" >>"$tmp"
          seen="$seen$k "
          replaced=true
          break
        fi
      done
      if [[ "$replaced" == false ]]; then
        printf '%s\n' "$line" >>"$tmp"
      fi
    done <"$file"
  fi

  for k in "${keys[@]}"; do
    if [[ "$seen" != *" $k "* ]]; then
      printf '%s=%s\n' "$k" "${!k-}" >>"$tmp"
    fi
  done

  mv "$tmp" "$file"
}

upsert_env "$ENV_FILE" \
  MARV_CONFIG_DIR \
  MARV_WORKSPACE_DIR \
  MARV_GATEWAY_PORT \
  MARV_BRIDGE_PORT \
  MARV_GATEWAY_BIND \
  MARV_GATEWAY_TOKEN \
  MARV_IMAGE \
  MARV_EXTRA_MOUNTS \
  MARV_HOME_VOLUME \
  MARV_DOCKER_APT_PACKAGES

echo "==> Building Docker image: $IMAGE_NAME"
docker build \
  --build-arg "MARV_DOCKER_APT_PACKAGES=${MARV_DOCKER_APT_PACKAGES}" \
  -t "$IMAGE_NAME" \
  -f "$ROOT_DIR/Dockerfile" \
  "$ROOT_DIR"

echo ""
echo "==> Onboarding (interactive)"
echo "When prompted:"
echo "  - Gateway bind: lan"
echo "  - Gateway auth: token"
echo "  - Gateway token: $MARV_GATEWAY_TOKEN"
echo "  - Tailscale exposure: Off"
echo "  - Install Gateway daemon: No"
echo ""
docker compose "${COMPOSE_ARGS[@]}" run --rm marv-cli onboard --no-install-daemon

echo ""
echo "==> Provider setup (optional)"
echo "WhatsApp (QR):"
echo "  ${COMPOSE_HINT} run --rm marv-cli channels login"
echo "Telegram (bot token):"
echo "  ${COMPOSE_HINT} run --rm marv-cli channels add --channel telegram --token <token>"
echo "Discord (bot token):"
echo "  ${COMPOSE_HINT} run --rm marv-cli channels add --channel discord --token <token>"
echo "Docs: /channels"

echo ""
echo "==> Starting gateway"
docker compose "${COMPOSE_ARGS[@]}" up -d marv-gateway

echo ""
echo "Gateway running with host port mapping."
echo "Access from tailnet devices via the host's tailnet IP."
echo "Config: $MARV_CONFIG_DIR"
echo "Workspace: $MARV_WORKSPACE_DIR"
echo "Token: $MARV_GATEWAY_TOKEN"
echo ""
echo "Commands:"
echo "  ${COMPOSE_HINT} logs -f marv-gateway"
echo "  ${COMPOSE_HINT} exec marv-gateway node dist/index.js health --token \"$MARV_GATEWAY_TOKEN\""

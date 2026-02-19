#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required for scripts/telegram_live_check.sh" >&2
  exit 1
fi

EDGE_BASE_URL="${EDGE_BASE_URL:-http://127.0.0.1:8000}"
TELEGRAM_API_BASE_URL="${TELEGRAM_API_BASE_URL:-https://api.telegram.org}"
TELEGRAM_UPDATES_LIMIT="${TELEGRAM_UPDATES_LIMIT:-10}"
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"

if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
  echo "TELEGRAM_BOT_TOKEN is required." >&2
  exit 1
fi

telegram_pid_file="$ROOT_DIR/.run/telegram.pid"
telegram_running=false
telegram_pid=
if [ -f "$telegram_pid_file" ]; then
  telegram_pid="$(cat "$telegram_pid_file")"
  if kill -0 "$telegram_pid" >/dev/null 2>&1; then
    telegram_running=true
  fi
fi

echo "[check] edge health"
edge_health="$(curl -fsS "$EDGE_BASE_URL/health")"

echo "[check] telegram getMe"
bot_info="$(curl -fsS "$TELEGRAM_API_BASE_URL/bot$TELEGRAM_BOT_TOKEN/getMe")"
bot_ok="$(echo "$bot_info" | jq -r '.ok')"
if [ "$bot_ok" != "true" ]; then
  echo "telegram getMe failed: $bot_info" >&2
  exit 1
fi

echo "[check] telegram getUpdates"
updates="$(curl -fsS "$TELEGRAM_API_BASE_URL/bot$TELEGRAM_BOT_TOKEN/getUpdates?limit=$TELEGRAM_UPDATES_LIMIT")"

summary="$(jq -n \
  --argjson edge_health "$edge_health" \
  --argjson bot_info "$bot_info" \
  --argjson updates "$updates" \
  --arg telegram_pid_file "$telegram_pid_file" \
  --arg telegram_pid "$telegram_pid" \
  --argjson telegram_running "$telegram_running" \
  '{
    edge_health: $edge_health,
    telegram_process: {
      pid_file: $telegram_pid_file,
      pid: (if $telegram_pid == "" then null else ($telegram_pid | tonumber) end),
      running: $telegram_running
    },
    bot: {
      id: $bot_info.result.id,
      username: $bot_info.result.username,
      can_join_groups: $bot_info.result.can_join_groups
    },
    updates: {
      count: ($updates.result | length),
      latest: (
        if ($updates.result | length) == 0
        then null
        else (
          $updates.result[-1]
          | {
              update_id,
              chat_id: (.message.chat.id // null),
              from_id: (.message.from.id // null),
              text: (.message.text // null),
              date: (.message.date // null)
            }
        )
        end
      )
    }
  }')"

echo "$summary"

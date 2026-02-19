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
  echo "jq is required for scripts/live_tuning.sh" >&2
  exit 1
fi

EDGE_BASE_URL="${EDGE_BASE_URL:-http://127.0.0.1:8000}"
TUNE_ACTOR_ID="${TUNE_ACTOR_ID:-tune-owner}"
TUNE_ACTOR_ROLE="${TUNE_ACTOR_ROLE:-owner}"
TUNE_CHANNEL="${TUNE_CHANNEL:-telegram}"
TUNE_CHANNEL_ID="${TUNE_CHANNEL_ID:-${TELEGRAM_CHAT_ID:-}}"
TUNE_USER_ID="${TUNE_USER_ID:-${TELEGRAM_USER_ID:-tune-user}}"
TUNE_THREAD_ID="${TUNE_THREAD_ID:-}"
TUNE_RUNS="${TUNE_RUNS:-3}"
TUNE_TIMEOUT_SECONDS="${TUNE_TIMEOUT_SECONDS:-90}"
TUNE_POLL_INTERVAL_SECONDS="${TUNE_POLL_INTERVAL_SECONDS:-0.5}"
TUNE_MESSAGE_PREFIX="${TUNE_MESSAGE_PREFIX:-runtime tune probe}"
TUNE_OUTPUT_DIR="${TUNE_OUTPUT_DIR:-$ROOT_DIR/logs}"
TUNE_STYLE_TEXT="${TUNE_STYLE_TEXT:-}"
TUNE_HEARTBEAT_INTERVAL_SECONDS="${TUNE_HEARTBEAT_INTERVAL_SECONDS:-}"
TUNE_CONVERSATION_ID="${TUNE_CONVERSATION_ID:-}"

if [ -z "$TUNE_CHANNEL_ID" ]; then
  echo "TUNE_CHANNEL_ID is required (or set TELEGRAM_CHAT_ID in .env)." >&2
  exit 1
fi

if [ -z "$TUNE_CONVERSATION_ID" ]; then
  if [ "$TUNE_CHANNEL" = "telegram" ]; then
    TUNE_CONVERSATION_ID="telegram:${TUNE_CHANNEL_ID}:0"
  else
    TUNE_CONVERSATION_ID="tune:${TUNE_CHANNEL}:${TUNE_CHANNEL_ID}"
  fi
fi

mkdir -p "$TUNE_OUTPUT_DIR"
OUTPUT_FILE="$TUNE_OUTPUT_DIR/live_tuning_$(date -u +%Y%m%d_%H%M%S).jsonl"

owner_args=(
  --edge-base-url "$EDGE_BASE_URL"
  --actor-id "$TUNE_ACTOR_ID"
  --actor-role "$TUNE_ACTOR_ROLE"
)

echo "[tuning] edge health check"
uv run marv "${owner_args[@]}" health >/dev/null

if [ -n "$TUNE_HEARTBEAT_INTERVAL_SECONDS" ]; then
  echo "[tuning] set heartbeat interval=${TUNE_HEARTBEAT_INTERVAL_SECONDS}s"
  uv run marv "${owner_args[@]}" heartbeat set --mode interval --interval-seconds "$TUNE_HEARTBEAT_INTERVAL_SECONDS" >/dev/null
fi

if [ -n "$TUNE_STYLE_TEXT" ]; then
  echo "[tuning] apply channel patch: $TUNE_STYLE_TEXT"
  proposal="$(uv run marv "${owner_args[@]}" config propose --text "$TUNE_STYLE_TEXT" --scope-type channel --scope-id "${TUNE_CHANNEL}:${TUNE_CHANNEL_ID}")"
  proposal_id="$(echo "$proposal" | jq -r '.proposal_id // empty')"
  if [ -z "$proposal_id" ]; then
    echo "failed to create proposal: $proposal" >&2
    exit 1
  fi
  uv run marv "${owner_args[@]}" config commit "$proposal_id" >/dev/null
fi

echo "[tuning] output file: $OUTPUT_FILE"
for i in $(seq 1 "$TUNE_RUNS"); do
  probe_message="${TUNE_MESSAGE_PREFIX} #${i} @$(date -u +%H:%M:%S)"
  cmd=(
    uv run marv "${owner_args[@]}"
    ops probe
    --message "$probe_message"
    --conversation-id "$TUNE_CONVERSATION_ID"
    --channel "$TUNE_CHANNEL"
    --channel-id "$TUNE_CHANNEL_ID"
    --user-id "$TUNE_USER_ID"
    --timeout-seconds "$TUNE_TIMEOUT_SECONDS"
    --poll-interval-seconds "$TUNE_POLL_INTERVAL_SECONDS"
  )
  if [ -n "$TUNE_THREAD_ID" ]; then
    cmd+=(--thread-id "$TUNE_THREAD_ID")
  fi

  probe_json="$("${cmd[@]}")"
  echo "$probe_json" | jq --argjson run "$i" --arg probe_message "$probe_message" '. + {run: $run, probe_message: $probe_message}' >> "$OUTPUT_FILE"

  status="$(echo "$probe_json" | jq -r '.status')"
  wall_ms="$(echo "$probe_json" | jq -r '.wall_time_ms')"
  event_ms="$(echo "$probe_json" | jq -r '.event_latency_ms')"
  echo "[run $i/$TUNE_RUNS] status=$status wall=${wall_ms}ms event=${event_ms}ms"
done

avg_wall="$(jq -s 'map(.wall_time_ms // 0) | if length == 0 then 0 else (add / length | floor) end' "$OUTPUT_FILE")"
avg_event="$(jq -s 'map(.event_latency_ms // 0) | if length == 0 then 0 else (add / length | floor) end' "$OUTPUT_FILE")"
echo "[tuning] avg wall=${avg_wall}ms avg event=${avg_event}ms"
echo "[tuning] done"

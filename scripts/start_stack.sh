#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
PID_DIR="$ROOT_DIR/.run"

mkdir -p "$LOG_DIR" "$PID_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

stop_if_running() {
  local pid_file="$1"
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$pid_file"
  fi
}

stop_if_running "$PID_DIR/core.pid"
stop_if_running "$PID_DIR/edge.pid"
stop_if_running "$PID_DIR/telegram.pid"

cd "$ROOT_DIR"

uv run uvicorn core.api:app --host 127.0.0.1 --port 9000 > "$LOG_DIR/core.log" 2>&1 &
echo $! > "$PID_DIR/core.pid"

CORE_BASE_URL=http://127.0.0.1:9000 uv run uvicorn backend.agent.api:app --host 127.0.0.1 --port 8000 > "$LOG_DIR/edge.log" 2>&1 &
echo $! > "$PID_DIR/edge.pid"

echo "core pid: $(cat "$PID_DIR/core.pid")"
echo "edge pid: $(cat "$PID_DIR/edge.pid")"

if [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
  EDGE_BASE_URL=http://127.0.0.1:8000 uv run marv-telegram > "$LOG_DIR/telegram.log" 2>&1 &
  echo $! > "$PID_DIR/telegram.pid"
  echo "telegram pid: $(cat "$PID_DIR/telegram.pid")"
else
  echo "telegram disabled: set TELEGRAM_BOT_TOKEN to enable"
fi

echo "logs: $LOG_DIR"

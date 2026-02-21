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

stop_if_running "$PID_DIR/frontend.pid"
stop_if_running "$PID_DIR/telegram.pid"
stop_if_running "$PID_DIR/edge.pid"
stop_if_running "$PID_DIR/core.pid"

check_port_free() {
  local port="$1"
  local name="$2"
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "cannot start $name: port $port already in use"
    lsof -nP -iTCP:"$port" -sTCP:LISTEN || true
    exit 1
  fi
}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found in PATH"
  exit 1
fi

check_port_free 9000 core
check_port_free 8000 edge

start_frontend="${MARV_START_FRONTEND:-true}"
if [ "$start_frontend" = "true" ]; then
  check_port_free 3000 frontend
fi

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

if [ "$start_frontend" = "true" ]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "frontend skipped: npm not found in PATH"
  else
    if [ ! -d "$ROOT_DIR/frontend/node_modules" ] && [ "${MARV_INSTALL_FRONTEND_DEPS:-true}" = "true" ]; then
      echo "installing frontend dependencies..."
      (cd "$ROOT_DIR/frontend" && npm ci > "$LOG_DIR/frontend_install.log" 2>&1)
    fi
    if [ -d "$ROOT_DIR/frontend/node_modules" ]; then
      cd "$ROOT_DIR/frontend"
      NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-http://127.0.0.1:8000}" npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
      echo $! > "$PID_DIR/frontend.pid"
      cd "$ROOT_DIR"
      echo "frontend pid: $(cat "$PID_DIR/frontend.pid")"
    else
      echo "frontend skipped: dependencies missing (run cd frontend && npm ci)"
    fi
  fi
else
  echo "frontend disabled: set MARV_START_FRONTEND=true to enable"
fi

echo "core url: http://127.0.0.1:9000"
echo "edge url: http://127.0.0.1:8000"
if [ -f "$PID_DIR/frontend.pid" ]; then
  echo "console url: http://127.0.0.1:3000/chat"
fi
echo "logs: $LOG_DIR"

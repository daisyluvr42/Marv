#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/daisyluvr42/Marv.git}"
TARGET_DIR="${TARGET_DIR:-$HOME/Marv}"
BRANCH="${BRANCH:-main}"

info() { printf "[bootstrap] %s\n" "$*"; }
fail() { printf "[bootstrap][error] %s\n" "$*" >&2; exit 1; }

if ! command -v git >/dev/null 2>&1; then
  fail "git 未安装，请先安装 Xcode Command Line Tools（xcode-select --install）。"
fi

if [ -d "$TARGET_DIR/.git" ]; then
  info "更新已有仓库: $TARGET_DIR"
  git -C "$TARGET_DIR" fetch origin "$BRANCH"
  git -C "$TARGET_DIR" checkout "$BRANCH"
  git -C "$TARGET_DIR" pull --ff-only origin "$BRANCH"
else
  info "克隆仓库到: $TARGET_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"

if ! command -v uv >/dev/null 2>&1; then
  fail "uv 未安装。建议执行: brew install uv"
fi

if ! command -v node >/dev/null 2>&1; then
  fail "node 未安装。建议通过 nvm 安装 20.11.1"
fi

NODE_VERSION_RAW="$(node -v | sed 's/^v//')"
NODE_MAJOR="${NODE_VERSION_RAW%%.*}"
if [ "$NODE_MAJOR" -lt 20 ]; then
  fail "当前 Node 版本过低: v$NODE_VERSION_RAW。请使用 .nvmrc 指定版本（20.11.1）。"
fi

if ! command -v npm >/dev/null 2>&1; then
  fail "npm 未安装。"
fi

info "同步 Python 依赖"
uv sync

info "安装前端依赖"
(cd frontend && npm ci)

info "运行后端测试"
uv run pytest -q

info "构建前端"
(cd frontend && npm run build)

cat <<'MSG'

[bootstrap] 完成。

启动命令：
1) uv run uvicorn core.api:app --port 9000
2) CORE_BASE_URL=http://127.0.0.1:9000 uv run uvicorn backend.agent.api:app --port 8000
3) cd frontend && NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev

冒烟测试：
EDGE_BASE_URL=http://127.0.0.1:8000 bash scripts/smoke_test.sh

MSG

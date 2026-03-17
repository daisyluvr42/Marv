#!/usr/bin/env bash
# MarvDock - Docker helpers for Marv
# Inspired by Simon Willison's "Running Marv in Docker"
# https://til.simonwillison.net/llms/marv-docker
#
# Installation:
#   mkdir -p ~/.marvdock && curl -sL https://raw.githubusercontent.com/marv/marv/main/scripts/shell-helpers/marvdock-helpers.sh -o ~/.marvdock/marvdock-helpers.sh
#   echo 'source ~/.marvdock/marvdock-helpers.sh' >> ~/.zshrc
#
# Usage:
#   marvdock-help    # Show all available commands

# =============================================================================
# Colors
# =============================================================================
_CLR_RESET='\033[0m'
_CLR_BOLD='\033[1m'
_CLR_DIM='\033[2m'
_CLR_GREEN='\033[0;32m'
_CLR_YELLOW='\033[1;33m'
_CLR_BLUE='\033[0;34m'
_CLR_MAGENTA='\033[0;35m'
_CLR_CYAN='\033[0;36m'
_CLR_RED='\033[0;31m'

# Styled command output (green + bold)
_clr_cmd() {
  echo -e "${_CLR_GREEN}${_CLR_BOLD}$1${_CLR_RESET}"
}

# Inline command for use in sentences
_cmd() {
  echo "${_CLR_GREEN}${_CLR_BOLD}$1${_CLR_RESET}"
}

# =============================================================================
# Config
# =============================================================================
MARVDOCK_CONFIG="${HOME}/.marvdock/config"

# Common paths to check for Marv
MARVDOCK_COMMON_PATHS=(
  "${HOME}/marv"
  "${HOME}/workspace/marv"
  "${HOME}/projects/marv"
  "${HOME}/dev/marv"
  "${HOME}/code/marv"
  "${HOME}/src/marv"
)

_marvdock_filter_warnings() {
  grep -v "^WARN\|^time="
}

_marvdock_trim_quotes() {
  local value="$1"
  value="${value#\"}"
  value="${value%\"}"
  printf "%s" "$value"
}

_marvdock_read_config_dir() {
  if [[ ! -f "$MARVDOCK_CONFIG" ]]; then
    return 1
  fi
  local raw
  raw=$(sed -n 's/^MARVDOCK_DIR=//p' "$MARVDOCK_CONFIG" | head -n 1)
  if [[ -z "$raw" ]]; then
    return 1
  fi
  _marvdock_trim_quotes "$raw"
}

# Ensure MARVDOCK_DIR is set and valid
_marvdock_ensure_dir() {
  # Already set and valid?
  if [[ -n "$MARVDOCK_DIR" && -f "${MARVDOCK_DIR}/docker-compose.yml" ]]; then
    return 0
  fi

  # Try loading from config
  local config_dir
  config_dir=$(_marvdock_read_config_dir)
  if [[ -n "$config_dir" && -f "${config_dir}/docker-compose.yml" ]]; then
    MARVDOCK_DIR="$config_dir"
    return 0
  fi

  # Auto-detect from common paths
  local found_path=""
  for path in "${MARVDOCK_COMMON_PATHS[@]}"; do
    if [[ -f "${path}/docker-compose.yml" ]]; then
      found_path="$path"
      break
    fi
  done

  if [[ -n "$found_path" ]]; then
    echo ""
    echo "🤖 Found Marv at: $found_path"
    echo -n "   Use this location? [Y/n] "
    read -r response
    if [[ "$response" =~ ^[Nn] ]]; then
      echo ""
      echo "Set MARVDOCK_DIR manually:"
      echo "  export MARVDOCK_DIR=/path/to/marv"
      return 1
    fi
    MARVDOCK_DIR="$found_path"
  else
    echo ""
    echo "❌ Marv not found in common locations."
    echo ""
    echo "Clone it first:"
    echo ""
    echo "  git clone https://github.com/daisyluvr42/Marv.git ~/marv"
    echo "  cd ~/marv && ./docker-setup.sh"
    echo ""
    echo "Or set MARVDOCK_DIR if it's elsewhere:"
    echo ""
    echo "  export MARVDOCK_DIR=/path/to/marv"
    echo ""
    return 1
  fi

  # Save to config
  if [[ ! -d "${HOME}/.marvdock" ]]; then
    /bin/mkdir -p "${HOME}/.marvdock"
  fi
  echo "MARVDOCK_DIR=\"$MARVDOCK_DIR\"" > "$MARVDOCK_CONFIG"
  echo "✅ Saved to $MARVDOCK_CONFIG"
  echo ""
  return 0
}

# Wrapper to run docker compose commands
_marvdock_compose() {
  _marvdock_ensure_dir || return 1
  command docker compose -f "${MARVDOCK_DIR}/docker-compose.yml" "$@"
}

_marvdock_read_env_token() {
  _marvdock_ensure_dir || return 1
  if [[ ! -f "${MARVDOCK_DIR}/.env" ]]; then
    return 1
  fi
  local raw
  raw=$(sed -n 's/^MARV_GATEWAY_TOKEN=//p' "${MARVDOCK_DIR}/.env" | head -n 1)
  if [[ -z "$raw" ]]; then
    return 1
  fi
  _marvdock_trim_quotes "$raw"
}

# Basic Operations
marvdock-start() {
  _marvdock_compose up -d marv-gateway
}

marvdock-stop() {
  _marvdock_compose down
}

marvdock-restart() {
  _marvdock_compose restart marv-gateway
}

marvdock-logs() {
  _marvdock_compose logs -f marv-gateway
}

marvdock-status() {
  _marvdock_compose ps
}

# Navigation
marvdock-cd() {
  _marvdock_ensure_dir || return 1
  cd "${MARVDOCK_DIR}"
}

marvdock-config() {
  cd ~/.marv
}

marvdock-workspace() {
  cd ~/.marv/workspace
}

# Container Access
marvdock-shell() {
  _marvdock_compose exec marv-gateway \
    bash -c 'echo "alias marv=\"./marv.mjs\"" > /tmp/.bashrc_marv && bash --rcfile /tmp/.bashrc_marv'
}

marvdock-exec() {
  _marvdock_compose exec marv-gateway "$@"
}

marvdock-cli() {
  _marvdock_compose run --rm marv-cli "$@"
}

# Maintenance
marvdock-rebuild() {
  _marvdock_compose build marv-gateway
}

marvdock-clean() {
  _marvdock_compose down -v --remove-orphans
}

# Health check
marvdock-health() {
  _marvdock_ensure_dir || return 1
  local token
  token=$(_marvdock_read_env_token)
  if [[ -z "$token" ]]; then
    echo "❌ Error: Could not find gateway token"
    echo "   Check: ${MARVDOCK_DIR}/.env"
    return 1
  fi
  _marvdock_compose exec -e "MARV_GATEWAY_TOKEN=$token" marv-gateway \
    node dist/index.js health
}

# Show gateway token
marvdock-token() {
  _marvdock_read_env_token
}

# Fix token configuration (run this once after setup)
marvdock-fix-token() {
  _marvdock_ensure_dir || return 1

  echo "🔧 Configuring gateway token..."
  local token
  token=$(marvdock-token)
  if [[ -z "$token" ]]; then
    echo "❌ Error: Could not find gateway token"
    echo "   Check: ${MARVDOCK_DIR}/.env"
    return 1
  fi

  echo "📝 Setting token: ${token:0:20}..."

  _marvdock_compose exec -e "TOKEN=$token" marv-gateway \
    bash -c './marv.mjs config set gateway.remote.token "$TOKEN" && ./marv.mjs config set gateway.auth.token "$TOKEN"' 2>&1 | _marvdock_filter_warnings

  echo "🔍 Verifying token was saved..."
  local saved_token
  saved_token=$(_marvdock_compose exec marv-gateway \
    bash -c "./marv.mjs config get gateway.remote.token 2>/dev/null" 2>&1 | _marvdock_filter_warnings | tr -d '\r\n' | head -c 64)

  if [[ "$saved_token" == "$token" ]]; then
    echo "✅ Token saved correctly!"
  else
    echo "⚠️  Token mismatch detected"
    echo "   Expected: ${token:0:20}..."
    echo "   Got: ${saved_token:0:20}..."
  fi

  echo "🔄 Restarting gateway..."
  _marvdock_compose restart marv-gateway 2>&1 | _marvdock_filter_warnings

  echo "⏳ Waiting for gateway to start..."
  sleep 5

  echo "✅ Configuration complete!"
  echo -e "   Try: $(_cmd marvdock-devices)"
}

# Open dashboard in browser
marvdock-dashboard() {
  _marvdock_ensure_dir || return 1

  echo "🤖 Getting dashboard URL..."
  local output exit_status url
  output=$(_marvdock_compose run --rm marv-cli dashboard --no-open 2>&1)
  exit_status=$?
  url=$(printf "%s\n" "$output" | _marvdock_filter_warnings | grep -o 'http[s]\?://[^[:space:]]*' | head -n 1)
  if [[ $exit_status -ne 0 ]]; then
    echo "❌ Failed to get dashboard URL"
    echo -e "   Try restarting: $(_cmd marvdock-restart)"
    return 1
  fi

  if [[ -n "$url" ]]; then
    echo "✅ Opening: $url"
    open "$url" 2>/dev/null || xdg-open "$url" 2>/dev/null || echo "   Please open manually: $url"
    echo ""
    echo -e "${_CLR_CYAN}💡 If you see 'pairing required' error:${_CLR_RESET}"
    echo -e "   1. Run: $(_cmd marvdock-devices)"
    echo "   2. Copy the Request ID from the Pending table"
    echo -e "   3. Run: $(_cmd 'marvdock-approve <request-id>')"
  else
    echo "❌ Failed to get dashboard URL"
    echo -e "   Try restarting: $(_cmd marvdock-restart)"
  fi
}

# List device pairings
marvdock-devices() {
  _marvdock_ensure_dir || return 1

  echo "🔍 Checking device pairings..."
  local output exit_status
  output=$(_marvdock_compose exec marv-gateway node dist/index.js devices list 2>&1)
  exit_status=$?
  printf "%s\n" "$output" | _marvdock_filter_warnings
  if [ $exit_status -ne 0 ]; then
    echo ""
    echo -e "${_CLR_CYAN}💡 If you see token errors above:${_CLR_RESET}"
    echo -e "   1. Verify token is set: $(_cmd marvdock-token)"
    echo "   2. Try manual config inside container:"
    echo -e "      $(_cmd marvdock-shell)"
    echo -e "      $(_cmd 'marv config get gateway.remote.token')"
    return 1
  fi

  echo ""
  echo -e "${_CLR_CYAN}💡 To approve a pairing request:${_CLR_RESET}"
  echo -e "   $(_cmd 'marvdock-approve <request-id>')"
}

# Approve device pairing request
marvdock-approve() {
  _marvdock_ensure_dir || return 1

  if [[ -z "$1" ]]; then
    echo -e "❌ Usage: $(_cmd 'marvdock-approve <request-id>')"
    echo ""
    echo -e "${_CLR_CYAN}💡 How to approve a device:${_CLR_RESET}"
    echo -e "   1. Run: $(_cmd marvdock-devices)"
    echo "   2. Find the Request ID in the Pending table (long UUID)"
    echo -e "   3. Run: $(_cmd 'marvdock-approve <that-request-id>')"
    echo ""
    echo "Example:"
    echo -e "   $(_cmd 'marvdock-approve 6f9db1bd-a1cc-4d3f-b643-2c195262464e')"
    return 1
  fi

  echo "✅ Approving device: $1"
  _marvdock_compose exec marv-gateway \
    node dist/index.js devices approve "$1" 2>&1 | _marvdock_filter_warnings

  echo ""
  echo "✅ Device approved! Refresh your browser."
}

# Show all available marvdock helper commands
marvdock-help() {
  echo -e "\n${_CLR_BOLD}${_CLR_CYAN}🤖 MarvDock - Docker Helpers for Marv${_CLR_RESET}\n"

  echo -e "${_CLR_BOLD}${_CLR_MAGENTA}⚡ Basic Operations${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-start)       ${_CLR_DIM}Start the gateway${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-stop)        ${_CLR_DIM}Stop the gateway${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-restart)     ${_CLR_DIM}Restart the gateway${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-status)      ${_CLR_DIM}Check container status${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-logs)        ${_CLR_DIM}View live logs (follows)${_CLR_RESET}"
  echo ""

  echo -e "${_CLR_BOLD}${_CLR_MAGENTA}🐚 Container Access${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-shell)       ${_CLR_DIM}Shell into container (marv alias ready)${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-cli)         ${_CLR_DIM}Run CLI commands (e.g., marvdock-cli status)${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-exec) ${_CLR_CYAN}<cmd>${_CLR_RESET}  ${_CLR_DIM}Execute command in gateway container${_CLR_RESET}"
  echo ""

  echo -e "${_CLR_BOLD}${_CLR_MAGENTA}🌐 Web UI & Devices${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-dashboard)   ${_CLR_DIM}Open web UI in browser ${_CLR_CYAN}(auto-guides you)${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-devices)     ${_CLR_DIM}List device pairings ${_CLR_CYAN}(auto-guides you)${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-approve) ${_CLR_CYAN}<id>${_CLR_RESET} ${_CLR_DIM}Approve device pairing ${_CLR_CYAN}(with examples)${_CLR_RESET}"
  echo ""

  echo -e "${_CLR_BOLD}${_CLR_MAGENTA}⚙️  Setup & Configuration${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-fix-token)   ${_CLR_DIM}Configure gateway token ${_CLR_CYAN}(run once)${_CLR_RESET}"
  echo ""

  echo -e "${_CLR_BOLD}${_CLR_MAGENTA}🔧 Maintenance${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-rebuild)     ${_CLR_DIM}Rebuild Docker image${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-clean)       ${_CLR_RED}⚠️  Remove containers & volumes (nuclear)${_CLR_RESET}"
  echo ""

  echo -e "${_CLR_BOLD}${_CLR_MAGENTA}🛠️  Utilities${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-health)      ${_CLR_DIM}Run health check${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-token)       ${_CLR_DIM}Show gateway auth token${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-cd)          ${_CLR_DIM}Jump to marv project directory${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-config)      ${_CLR_DIM}Open config directory (~/.marv)${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-workspace)   ${_CLR_DIM}Open workspace directory${_CLR_RESET}"
  echo ""

  echo -e "${_CLR_BOLD}${_CLR_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${_CLR_RESET}"
  echo -e "${_CLR_BOLD}${_CLR_GREEN}🚀 First Time Setup${_CLR_RESET}"
  echo -e "${_CLR_CYAN}  1.${_CLR_RESET} $(_cmd marvdock-start)          ${_CLR_DIM}# Start the gateway${_CLR_RESET}"
  echo -e "${_CLR_CYAN}  2.${_CLR_RESET} $(_cmd marvdock-fix-token)      ${_CLR_DIM}# Configure token${_CLR_RESET}"
  echo -e "${_CLR_CYAN}  3.${_CLR_RESET} $(_cmd marvdock-dashboard)      ${_CLR_DIM}# Open web UI${_CLR_RESET}"
  echo -e "${_CLR_CYAN}  4.${_CLR_RESET} $(_cmd marvdock-devices)        ${_CLR_DIM}# If pairing needed${_CLR_RESET}"
  echo -e "${_CLR_CYAN}  5.${_CLR_RESET} $(_cmd marvdock-approve) ${_CLR_CYAN}<id>${_CLR_RESET}   ${_CLR_DIM}# Approve pairing${_CLR_RESET}"
  echo ""

  echo -e "${_CLR_BOLD}${_CLR_GREEN}💬 WhatsApp Setup${_CLR_RESET}"
  echo -e "  $(_cmd marvdock-shell)"
  echo -e "    ${_CLR_BLUE}>${_CLR_RESET} $(_cmd 'marv channels login --channel whatsapp')"
  echo -e "    ${_CLR_BLUE}>${_CLR_RESET} $(_cmd 'marv status')"
  echo ""

  echo -e "${_CLR_BOLD}${_CLR_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${_CLR_RESET}"
  echo ""

  echo -e "${_CLR_CYAN}💡 All commands guide you through next steps!${_CLR_RESET}"
  echo -e "${_CLR_BLUE}📚 Docs: ${_CLR_RESET}${_CLR_CYAN}${_CLR_RESET}"
  echo ""
}

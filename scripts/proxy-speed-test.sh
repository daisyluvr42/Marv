#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/proxy-speed-test.sh [options]

Options:
  --proxy URL              HTTP/HTTPS proxy URL, e.g. http://127.0.0.1:7890
  --socks5 HOST:PORT       SOCKS5 proxy, e.g. 127.0.0.1:1080
  --only-proxy             Skip direct baseline and test proxy only
  --rounds N               Test rounds per mode (default: 3)
  --download-bytes N       Download payload size in bytes (default: 100000000)
  --upload-bytes N         Upload payload size in bytes (default: 50000000)
  --down-url URL           Download endpoint (default: https://speed.cloudflare.com/__down)
  --up-url URL             Upload endpoint (default: https://speed.cloudflare.com/__up)
  --timeout SEC            Max seconds per request (default: 120)
  --connect-timeout SEC    Connect timeout seconds (default: 10)
  -h, --help               Show help

Examples:
  scripts/proxy-speed-test.sh --proxy http://127.0.0.1:7890
  scripts/proxy-speed-test.sh --socks5 127.0.0.1:1080 --rounds 5
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: missing command '$cmd'" >&2
    exit 1
  fi
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
    echo "Error: $name must be a positive integer, got '$value'" >&2
    exit 1
  fi
}

append_bytes_param() {
  local url="$1"
  local bytes="$2"
  if [[ "$url" == *\?* ]]; then
    echo "${url}&bytes=${bytes}"
  else
    echo "${url}?bytes=${bytes}"
  fi
}

to_mbps() {
  local bps="$1"
  awk -v b="$bps" 'BEGIN { printf "%.2f", (b * 8) / 1000000 }'
}

avg_values() {
  if [ "$#" -eq 0 ]; then
    echo ""
    return
  fi
  printf '%s\n' "$@" | awk '{ s += $1 } END { if (NR > 0) printf "%.6f", s / NR }'
}

median_values() {
  if [ "$#" -eq 0 ]; then
    echo ""
    return
  fi
  printf '%s\n' "$@" | sort -n | awk '
    { a[NR] = $1 }
    END {
      if (NR == 0) {
        print ""
      } else if (NR % 2 == 1) {
        printf "%.6f", a[(NR + 1) / 2]
      } else {
        printf "%.6f", (a[NR / 2] + a[NR / 2 + 1]) / 2
      }
    }
  '
}

min_values() {
  if [ "$#" -eq 0 ]; then
    echo ""
    return
  fi
  printf '%s\n' "$@" | sort -n | head -n 1
}

max_values() {
  if [ "$#" -eq 0 ]; then
    echo ""
    return
  fi
  printf '%s\n' "$@" | sort -n | tail -n 1
}

print_metric_summary() {
  local label="$1"
  shift
  if [ "$#" -eq 0 ]; then
    echo "$label: no successful samples"
    return
  fi
  local avg median min max
  avg="$(avg_values "$@")"
  median="$(median_values "$@")"
  min="$(min_values "$@")"
  max="$(max_values "$@")"
  echo "$label: samples=$# median=$(to_mbps "$median") Mbps avg=$(to_mbps "$avg") Mbps min=$(to_mbps "$min") Mbps max=$(to_mbps "$max") Mbps"
}

measure_download() {
  local mode="$1"
  local url="$2"
  local -a cmd
  cmd=(
    curl
    -L
    --silent
    --show-error
    --output /dev/null
    --max-time "$TIMEOUT"
    --connect-timeout "$CONNECT_TIMEOUT"
    --write-out "%{speed_download}|%{time_connect}|%{time_starttransfer}|%{time_total}"
  )

  if [ "$mode" = "direct" ]; then
    cmd+=(--noproxy "*")
  elif [ -n "$PROXY_URL" ]; then
    cmd+=(-x "$PROXY_URL")
  else
    cmd+=(--socks5-hostname "$SOCKS5_ADDR")
  fi

  "${cmd[@]}" "$url"
}

measure_upload() {
  local mode="$1"
  local url="$2"
  local -a cmd
  cmd=(
    curl
    -L
    --silent
    --show-error
    --output /dev/null
    --max-time "$TIMEOUT"
    --connect-timeout "$CONNECT_TIMEOUT"
    --request POST
    --data-binary @-
    --write-out "%{speed_upload}|%{time_total}"
  )

  if [ "$mode" = "direct" ]; then
    cmd+=(--noproxy "*")
  elif [ -n "$PROXY_URL" ]; then
    cmd+=(-x "$PROXY_URL")
  else
    cmd+=(--socks5-hostname "$SOCKS5_ADDR")
  fi

  head -c "$UPLOAD_BYTES" /dev/zero | "${cmd[@]}" "$url"
}

ROUNDS=3
DOWNLOAD_BYTES=100000000
UPLOAD_BYTES=50000000
DOWN_URL="https://speed.cloudflare.com/__down"
UP_URL="https://speed.cloudflare.com/__up"
TIMEOUT=120
CONNECT_TIMEOUT=10
PROXY_URL=""
SOCKS5_ADDR=""
ONLY_PROXY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --proxy)
      PROXY_URL="${2:-}"
      shift 2
      ;;
    --socks5)
      SOCKS5_ADDR="${2:-}"
      shift 2
      ;;
    --only-proxy)
      ONLY_PROXY=1
      shift
      ;;
    --rounds)
      ROUNDS="${2:-}"
      shift 2
      ;;
    --download-bytes)
      DOWNLOAD_BYTES="${2:-}"
      shift 2
      ;;
    --upload-bytes)
      UPLOAD_BYTES="${2:-}"
      shift 2
      ;;
    --down-url)
      DOWN_URL="${2:-}"
      shift 2
      ;;
    --up-url)
      UP_URL="${2:-}"
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:-}"
      shift 2
      ;;
    --connect-timeout)
      CONNECT_TIMEOUT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_cmd curl
require_positive_int "rounds" "$ROUNDS"
require_positive_int "download-bytes" "$DOWNLOAD_BYTES"
require_positive_int "upload-bytes" "$UPLOAD_BYTES"
require_positive_int "timeout" "$TIMEOUT"
require_positive_int "connect-timeout" "$CONNECT_TIMEOUT"

if [ -n "$PROXY_URL" ] && [ -n "$SOCKS5_ADDR" ]; then
  echo "Error: use either --proxy or --socks5, not both." >&2
  exit 1
fi

if [ "$ONLY_PROXY" -eq 1 ] && [ -z "$PROXY_URL" ] && [ -z "$SOCKS5_ADDR" ]; then
  echo "Error: --only-proxy requires --proxy or --socks5." >&2
  exit 1
fi

MODES=("direct")
if [ -n "$PROXY_URL" ] || [ -n "$SOCKS5_ADDR" ]; then
  MODES+=("proxy")
fi
if [ "$ONLY_PROXY" -eq 1 ]; then
  MODES=("proxy")
fi

DOWNLOAD_URL="$DOWN_URL"
if [[ "$DOWN_URL" == *"__down"* ]]; then
  DOWNLOAD_URL="$(append_bytes_param "$DOWN_URL" "$DOWNLOAD_BYTES")"
fi

DIRECT_DL_BPS=()
DIRECT_UL_BPS=()
PROXY_DL_BPS=()
PROXY_UL_BPS=()
FAILURES=0

echo "=== Proxy Speed Test ==="
echo "Rounds per mode: $ROUNDS"
echo "Download bytes:  $DOWNLOAD_BYTES"
echo "Upload bytes:    $UPLOAD_BYTES"
echo "Download URL:    $DOWNLOAD_URL"
echo "Upload URL:      $UP_URL"
if [ -n "$PROXY_URL" ]; then
  echo "Proxy mode:      HTTP/HTTPS via $PROXY_URL"
elif [ -n "$SOCKS5_ADDR" ]; then
  echo "Proxy mode:      SOCKS5 via $SOCKS5_ADDR"
fi
echo

for mode in "${MODES[@]}"; do
  for ((round = 1; round <= ROUNDS; round++)); do
    if dl_raw="$(measure_download "$mode" "$DOWNLOAD_URL" 2>&1)"; then
      IFS='|' read -r dl_bps dl_connect dl_ttfb dl_total <<<"$dl_raw"
      printf '[Round %d][%s] Download: %s Mbps (connect=%ss, ttfb=%ss, total=%ss)\n' \
        "$round" "$mode" "$(to_mbps "$dl_bps")" "$dl_connect" "$dl_ttfb" "$dl_total"
      if [ "$mode" = "direct" ]; then
        DIRECT_DL_BPS+=("$dl_bps")
      else
        PROXY_DL_BPS+=("$dl_bps")
      fi
    else
      echo "[Round $round][$mode] Download failed: $dl_raw" >&2
      FAILURES=$((FAILURES + 1))
    fi

    if ul_raw="$(measure_upload "$mode" "$UP_URL" 2>&1)"; then
      IFS='|' read -r ul_bps ul_total <<<"$ul_raw"
      printf '[Round %d][%s] Upload:   %s Mbps (total=%ss)\n' \
        "$round" "$mode" "$(to_mbps "$ul_bps")" "$ul_total"
      if [ "$mode" = "direct" ]; then
        DIRECT_UL_BPS+=("$ul_bps")
      else
        PROXY_UL_BPS+=("$ul_bps")
      fi
    else
      echo "[Round $round][$mode] Upload failed: $ul_raw" >&2
      FAILURES=$((FAILURES + 1))
    fi
  done
done

echo
echo "=== Summary ==="

if [ "$ONLY_PROXY" -eq 0 ]; then
  echo "[direct]"
  if [ "${#DIRECT_DL_BPS[@]}" -gt 0 ]; then
    print_metric_summary "Download" "${DIRECT_DL_BPS[@]}"
  else
    print_metric_summary "Download"
  fi
  if [ "${#DIRECT_UL_BPS[@]}" -gt 0 ]; then
    print_metric_summary "Upload" "${DIRECT_UL_BPS[@]}"
  else
    print_metric_summary "Upload"
  fi
  echo
fi

if [ -n "$PROXY_URL" ] || [ -n "$SOCKS5_ADDR" ] || [ "$ONLY_PROXY" -eq 1 ]; then
  echo "[proxy]"
  if [ "${#PROXY_DL_BPS[@]}" -gt 0 ]; then
    print_metric_summary "Download" "${PROXY_DL_BPS[@]}"
  else
    print_metric_summary "Download"
  fi
  if [ "${#PROXY_UL_BPS[@]}" -gt 0 ]; then
    print_metric_summary "Upload" "${PROXY_UL_BPS[@]}"
  else
    print_metric_summary "Upload"
  fi
  echo
fi

if [ "$FAILURES" -gt 0 ]; then
  echo "Completed with $FAILURES failed request(s)." >&2
fi

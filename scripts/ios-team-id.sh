#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${IOS_DEVELOPMENT_TEAM:-}" ]]; then
  printf '%s\n' "${IOS_DEVELOPMENT_TEAM}"
  exit 0
fi

preferred_team="${IOS_PREFERRED_TEAM_ID:-${MARV_IOS_DEFAULT_TEAM_ID:-}}"
allow_keychain_fallback="${IOS_ALLOW_KEYCHAIN_TEAM_FALLBACK:-0}"
prefer_non_free_team="${IOS_PREFER_NON_FREE_TEAM:-1}"

declare -a team_ids=()
declare -a team_is_free=()

append_team() {
  local candidate_id="$1"
  local candidate_is_free="$2"
  [[ -z "$candidate_id" ]] && return

  local i
  for i in "${!team_ids[@]}"; do
    if [[ "${team_ids[$i]}" == "$candidate_id" ]]; then
      return
    fi
  done

  team_ids+=("$candidate_id")
  team_is_free+=("$candidate_is_free")
}

while IFS=$'\t' read -r team_id is_free; do
  [[ -z "$team_id" ]] && continue
  append_team "$team_id" "${is_free:-0}"
done < <(
  plutil -extract IDEProvisioningTeams json -o - "${HOME}/Library/Preferences/com.apple.dt.Xcode.plist" 2>/dev/null \
    | /usr/bin/python3 -c '
import json
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)

if not isinstance(data, dict):
    raise SystemExit(0)

for teams in data.values():
    if not isinstance(teams, list):
        continue
    for team in teams:
        if not isinstance(team, dict):
            continue
        team_id = str(team.get("teamID", "")).strip()
        if not team_id:
            continue
        is_free = "1" if bool(team.get("isFreeProvisioningTeam", False)) else "0"
        print(f"{team_id}\t{is_free}")
'
)

while IFS= read -r team; do
  [[ -z "$team" ]] && continue
  append_team "$team" "0"
done < <(
  defaults read com.apple.dt.Xcode IDEProvisioningTeamIdentifiers 2>/dev/null \
    | grep -Eo '[A-Z0-9]{10}' || true
)

if [[ ${#team_ids[@]} -eq 0 && "$allow_keychain_fallback" == "1" ]]; then
  while IFS= read -r team; do
    [[ -z "$team" ]] && continue
    append_team "$team" "0"
  done < <(
    security find-identity -p codesigning -v 2>/dev/null \
      | grep -Eo '\([A-Z0-9]{10}\)' \
      | tr -d '()' || true
  )
fi

if [[ ${#team_ids[@]} -eq 0 ]]; then
  echo "No Apple Team ID found in Xcode accounts. Open Xcode -> Settings -> Accounts and sign in, then retry." >&2
  echo "(Set IOS_ALLOW_KEYCHAIN_TEAM_FALLBACK=1 to allow keychain-only team detection.)" >&2
  exit 1
fi

if [[ -n "$preferred_team" ]]; then
  for i in "${!team_ids[@]}"; do
    if [[ "${team_ids[$i]}" == "$preferred_team" ]]; then
      printf '%s\n' "${team_ids[$i]}"
      exit 0
    fi
  done
fi

if [[ "$prefer_non_free_team" == "1" ]]; then
  for i in "${!team_ids[@]}"; do
    if [[ "${team_is_free[$i]}" == "0" ]]; then
      printf '%s\n' "${team_ids[$i]}"
      exit 0
    fi
  done
fi

printf '%s\n' "${team_ids[0]}"

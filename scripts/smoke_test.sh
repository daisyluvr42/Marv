#!/usr/bin/env bash
set -euo pipefail

EDGE_BASE_URL=${EDGE_BASE_URL:-http://127.0.0.1:8000}

owner_headers=(-H "X-Actor-Role: owner" -H "X-Actor-Id: smoke-owner")
json_headers=(-H "Content-Type: application/json")

echo "[smoke] health"
curl -fsS "$EDGE_BASE_URL/health" >/dev/null

echo "[smoke] agent message -> task"
msg=$(curl -fsS -X POST "$EDGE_BASE_URL/v1/agent/messages" "${json_headers[@]}" "${owner_headers[@]}" -d '{"message":"smoke hello","channel":"web"}')
task_id=$(echo "$msg" | jq -r '.task_id')
conv_id=$(echo "$msg" | jq -r '.conversation_id')

for _ in {1..30}; do
  task=$(curl -fsS "$EDGE_BASE_URL/v1/agent/tasks/$task_id")
  status=$(echo "$task" | jq -r '.status')
  if [[ "$status" == "completed" ]]; then
    break
  fi
  sleep 1
done

curl -fsS -X POST "$EDGE_BASE_URL/v1/audit/render" "${json_headers[@]}" -d "{\"task_id\":\"$task_id\"}" | jq '.summary' >/dev/null

echo "[smoke] tools read_only"
curl -fsS -X POST "$EDGE_BASE_URL/v1/tools:execute" "${json_headers[@]}" "${owner_headers[@]}" -d '{"tool":"mock_web_search","args":{"query":"smoke"}}' | jq '.status' >/dev/null

echo "[smoke] tools external_write + approval"
exec_out=$(curl -fsS -X POST "$EDGE_BASE_URL/v1/tools:execute" "${json_headers[@]}" "${owner_headers[@]}" -d '{"tool":"mock_external_write","args":{"target":"file://smoke","content":"payload"}}')
approval_id=$(echo "$exec_out" | jq -r '.approval_id')
curl -fsS -X POST "$EDGE_BASE_URL/v1/approvals/${approval_id}:approve" "${json_headers[@]}" "${owner_headers[@]}" -d '{}' | jq '.status' >/dev/null

echo "[smoke] config propose/commit"
proposal=$(curl -fsS -X POST "$EDGE_BASE_URL/v1/config/patches:propose" "${json_headers[@]}" "${owner_headers[@]}" -d '{"natural_language":"更简洁","scope_type":"channel","scope_id":"web:default"}')
proposal_id=$(echo "$proposal" | jq -r '.proposal_id')
curl -fsS -X POST "$EDGE_BASE_URL/v1/config/patches:commit" "${json_headers[@]}" "${owner_headers[@]}" -d "{\"proposal_id\":\"$proposal_id\"}" | jq '.status' >/dev/null

echo "[smoke] memory write/approve/query"
cand=$(curl -fsS -X POST "$EDGE_BASE_URL/v1/memory/write" "${json_headers[@]}" -d '{"scope_type":"user","scope_id":"smoke-user","kind":"preference","content":"我喜欢简洁回答","confidence":0.4,"requires_confirmation":true}')
cand_id=$(echo "$cand" | jq -r '.id')
curl -fsS -X POST "$EDGE_BASE_URL/v1/memory/candidates/${cand_id}:approve" "${owner_headers[@]}" | jq '.status' >/dev/null
curl -fsS -X POST "$EDGE_BASE_URL/v1/memory/query" "${json_headers[@]}" -d '{"scope_type":"user","scope_id":"smoke-user","query":"简洁回答","top_k":1}' | jq '.count' >/dev/null

echo "[smoke] done"

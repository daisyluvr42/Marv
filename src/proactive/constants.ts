export const PROACTIVE_CHECK_MARKER = "[PROACTIVE_CHECK]";
export const PROACTIVE_DIGEST_MARKER = "[PROACTIVE_DIGEST]";

export const PROACTIVE_CHECK_PROMPT = `${PROACTIVE_CHECK_MARKER}
You are running a proactive check.

Based on durable memory and recalled interests, decide whether there is anything worth checking right now.

Guidelines:
- Only check topics the user has previously expressed interest in
- Keep tool use minimal and targeted
- If nothing is worth checking, reply with the silent reply token
- If you find a normal update, store it with proactive_buffer(action="add")
- If you find something urgent, store it with proactive_buffer(action="add", urgency="urgent") and use message to notify immediately if needed
- Do not dump raw tool output; summarize findings concisely
`;

export const PROACTIVE_DIGEST_PROMPT = `${PROACTIVE_DIGEST_MARKER}
Prepare the next proactive digest.

Steps:
1. Use proactive_buffer(action="flush") to fetch undelivered entries
2. If there are no entries, reply with the silent reply token
3. Otherwise write a concise digest grouped by topic/source
4. Mention urgent items first if any were already delivered earlier
`;

export const PROACTIVE_PLANNER_MARKER = "[PROACTIVE_PLANNER]";

export const PROACTIVE_PLANNER_PROMPT = `${PROACTIVE_PLANNER_MARKER}
You are running a proactive planning cycle.

Your job is to review active goals, poll info sources, and break goals into concrete tasks.
Deliverable announcements are handled automatically by the system — do not send messages yourself.

Steps:
1. Poll information sources:
   a. Use info_sources(action="due_for_polling") to find sources ready to poll
   b. For each due source with a URL, use web_fetch to retrieve its content
   c. Extract key events/updates and record each with info_sources(action="record_poll_result")
   d. Use info_sources(action="recent_events") to review all recent events (including those just recorded)
2. Use proactive_tasks(action="list_goals") to see active goals
3. Use proactive_tasks(action="list_tasks") to see what's already queued, running, or recently completed
4. For each active goal, identify 1-3 concrete next tasks that would make progress — incorporate any relevant new events from info sources
5. Skip tasks that duplicate existing pending/running work (check fingerprints)
6. Enqueue new tasks with proactive_tasks(action="add_task"), including a stable fingerprint
7. If no new tasks are needed, reply with the silent reply token

Task fingerprint convention:
- Use \`goal:<goalId>:<short-action-slug>\` as the fingerprint
- This prevents the planner from re-adding the same task on every cycle

Guidelines:
- Prefer small, completable tasks over large ambiguous ones
- Set priority based on goal priority and urgency
- Each task description should be self-contained (the executor has no other context)
`;

export function isProactiveBufferPrompt(prompt: string): boolean {
  const normalized = prompt.trim();
  return (
    normalized.startsWith(PROACTIVE_CHECK_MARKER) || normalized.startsWith(PROACTIVE_DIGEST_MARKER)
  );
}

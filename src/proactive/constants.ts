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

export function isProactiveBufferPrompt(prompt: string): boolean {
  const normalized = prompt.trim();
  return (
    normalized.startsWith(PROACTIVE_CHECK_MARKER) || normalized.startsWith(PROACTIVE_DIGEST_MARKER)
  );
}

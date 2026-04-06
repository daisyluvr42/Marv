import { HEARTBEAT_TOKEN } from "../../auto-reply/support/tokens.js";

// Build a dynamic prompt for cron events by embedding the actual event content.
// This ensures the model sees the reminder text directly instead of relying on
// "shown in the system messages above" which may not be visible in context.
export function buildCronEventPrompt(pendingEvents: string[]): string {
  const eventText = pendingEvents.join("\n").trim();
  if (!eventText) {
    return (
      "A scheduled cron event was triggered, but no event content was found. " +
      "Reply HEARTBEAT_OK."
    );
  }
  return (
    "A scheduled reminder has been triggered. The reminder content is:\n\n" +
    eventText +
    "\n\nPlease relay this reminder to the user in a helpful and friendly way. " +
    "You MUST only restate or summarize the reminder content above. " +
    "Do NOT promise, claim, or suggest any follow-up actions (such as switching models, generating reports, sending messages, or retrying) unless you are actually executing them with a tool call in this same turn."
  );
}

function getHeartbeatOkPrefix(): string {
  return HEARTBEAT_TOKEN.toLowerCase();
}

// Detect heartbeat-specific noise so cron reminders don't trigger on non-reminder events.
function isHeartbeatAckEvent(evt: string): boolean {
  const trimmed = evt.trim();
  if (!trimmed) {
    return false;
  }
  const lower = trimmed.toLowerCase();
  const heartbeatOkPrefix = getHeartbeatOkPrefix();
  if (!lower.startsWith(heartbeatOkPrefix)) {
    return false;
  }
  const suffix = lower.slice(heartbeatOkPrefix.length);
  if (suffix.length === 0) {
    return true;
  }
  return !/[a-z0-9_]/.test(suffix[0]);
}

function isHeartbeatNoiseEvent(evt: string): boolean {
  const lower = evt.trim().toLowerCase();
  if (!lower) {
    return false;
  }
  return (
    isHeartbeatAckEvent(lower) ||
    lower.includes("heartbeat poll") ||
    lower.includes("heartbeat wake")
  );
}

export function isExecCompletionEvent(evt: string): boolean {
  return evt.toLowerCase().includes("exec finished");
}

// Returns true when a system event should be treated as real cron reminder content.
export function isCronSystemEvent(evt: string) {
  if (!evt.trim()) {
    return false;
  }
  return !isHeartbeatNoiseEvent(evt) && !isExecCompletionEvent(evt);
}

import { extractErrorCode, getErrorMessage, getStatusCode } from "../../infra/errors.js";
import { defaultVoiceWakeTriggers } from "../../infra/voicewake.js";

export function normalizeVoiceWakeTriggers(input: unknown): string[] {
  const raw = Array.isArray(input) ? input : [];
  const cleaned = raw
    .map((v) => (typeof v === "string" ? v.trim() : ""))
    .filter((v) => v.length > 0)
    .slice(0, 32)
    .map((v) => v.slice(0, 64));
  return cleaned.length > 0 ? cleaned : defaultVoiceWakeTriggers();
}

/** Gateway-specific error formatting that includes status/code when available. */
export function formatError(err: unknown): string {
  const msg = getErrorMessage(err);
  if (msg) {
    return msg;
  }
  const status = getStatusCode(err);
  const code = extractErrorCode(err);
  if (status !== undefined || code !== undefined) {
    return `status=${status ?? "unknown"} code=${code ?? "unknown"}`;
  }
  try {
    return JSON.stringify(err, null, 2);
  } catch {
    return String(err);
  }
}

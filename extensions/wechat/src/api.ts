import type { ResolvedWeChatAccount, WeChatProbeResult } from "./types.js";

/**
 * Probe the WeChat connection via Wechaty.
 */
export async function probeWeChat(account: ResolvedWeChatAccount): Promise<WeChatProbeResult> {
  if (!account.configured) {
    return { ok: false, error: "WeChat account not configured (no puppet specified)" };
  }
  // Actual probe implementation requires a running Wechaty instance.
  // This stub returns a basic status check.
  return { ok: true, loggedIn: false };
}

/**
 * Send a text message to a WeChat contact or room via Wechaty.
 */
export async function sendWeChatText(params: {
  account: ResolvedWeChatAccount;
  target: string;
  text: string;
}): Promise<void> {
  const { WechatyBuilder } = await import("wechaty" as string);
  // The actual send implementation requires a running bot instance.
  // In production, this would use the singleton Wechaty bot stored in runtime state.
  void WechatyBuilder; // Validate import availability.
  void params;
  throw new Error("WeChat send requires a running Wechaty bot instance. Start the gateway first.");
}

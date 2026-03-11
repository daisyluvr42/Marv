import type { MarvConfig } from "../core/config/config.js";
import { readDigestBuffer } from "./digest-buffer.js";

export type ProactiveStatusSnapshot = {
  agentId: string;
  enabled: boolean;
  checkEveryMinutes: number | null;
  digestTimes: string[];
  delivery: {
    channel: string;
    to: string | null;
  };
  totalEntries: number;
  pendingEntries: number;
  deliveredEntries: number;
  urgentEntries: number;
  lastFlushAt: number | null;
};

export async function getProactiveStatusSnapshot(params: {
  agentId: string;
  config?: MarvConfig;
}): Promise<ProactiveStatusSnapshot> {
  const cfg = params.config ?? {};
  const proactive = cfg.autonomy?.proactive;
  const buffer = await readDigestBuffer(params.agentId);
  const totalEntries = buffer.entries.length;
  const pendingEntries = buffer.entries.filter((entry) => !entry.delivered).length;
  const deliveredEntries = totalEntries - pendingEntries;
  const urgentEntries = buffer.entries.filter((entry) => entry.urgency === "urgent").length;
  return {
    agentId: params.agentId,
    enabled: proactive?.enabled === true,
    checkEveryMinutes:
      typeof proactive?.checkEveryMinutes === "number" &&
      Number.isFinite(proactive.checkEveryMinutes)
        ? Math.max(0, Math.floor(proactive.checkEveryMinutes))
        : null,
    digestTimes: Array.isArray(proactive?.digestTimes)
      ? proactive.digestTimes.filter(
          (value) => typeof value === "string" && value.trim().length > 0,
        )
      : [],
    delivery: {
      channel: proactive?.delivery?.channel?.trim() || "last",
      to: proactive?.delivery?.to?.trim() || null,
    },
    totalEntries,
    pendingEntries,
    deliveredEntries,
    urgentEntries,
    lastFlushAt: buffer.lastFlushAt > 0 ? buffer.lastFlushAt : null,
  };
}

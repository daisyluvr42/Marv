import { resolveDefaultAgentId } from "../../../../agents/agent-scope.js";
import type { loadConfig } from "../../../../core/config/config.js";
import type { resolveAgentRoute } from "../../../../routing/resolve-route.js";
import { buildAgentSessionKey } from "../../../../routing/resolve-route.js";
import {
  buildAgentMainSessionKey,
  DEFAULT_MAIN_KEY,
  normalizeAgentId,
} from "../../../../routing/session-key.js";
import { formatError } from "../../session.js";
import { whatsappInboundLog } from "../loggers.js";
import type { WebInboundMsg } from "../types.js";
import type { GroupHistoryEntry } from "./process-message.js";

export async function maybeBroadcastMessage(params: {
  cfg: ReturnType<typeof loadConfig>;
  msg: WebInboundMsg;
  peerId: string;
  route: ReturnType<typeof resolveAgentRoute>;
  groupHistoryKey: string;
  groupHistories: Map<string, GroupHistoryEntry[]>;
  processMessage: (
    msg: WebInboundMsg,
    route: ReturnType<typeof resolveAgentRoute>,
    groupHistoryKey: string,
    opts?: {
      groupHistory?: GroupHistoryEntry[];
      suppressGroupHistoryClear?: boolean;
    },
  ) => Promise<boolean>;
}) {
  const broadcastAgents = params.cfg.broadcast?.[params.peerId];
  if (!broadcastAgents || !Array.isArray(broadcastAgents)) {
    return false;
  }
  if (broadcastAgents.length === 0) {
    return false;
  }

  const strategy = params.cfg.broadcast?.strategy || "parallel";
  whatsappInboundLog.info(`Broadcasting message to ${broadcastAgents.length} agents (${strategy})`);

  const defaultAgentId = normalizeAgentId(resolveDefaultAgentId(params.cfg));
  const eligibleAgentIds = Array.from(
    new Set(
      broadcastAgents
        .map((agentId) => normalizeAgentId(agentId))
        .filter((agentId) => agentId === defaultAgentId),
    ),
  );
  if (eligibleAgentIds.length === 0) {
    whatsappInboundLog.warn("Broadcast config references only legacy agents; skipping broadcast.");
    return false;
  }
  const groupHistorySnapshot =
    params.msg.chatType === "group"
      ? (params.groupHistories.get(params.groupHistoryKey) ?? [])
      : undefined;

  const processForAgent = async (agentId: string): Promise<boolean> => {
    const normalizedAgentId = normalizeAgentId(agentId);
    const agentRoute = {
      ...params.route,
      agentId: normalizedAgentId,
      sessionKey: buildAgentSessionKey({
        agentId: normalizedAgentId,
        channel: "whatsapp",
        accountId: params.route.accountId,
        peer: {
          kind: params.msg.chatType === "group" ? "group" : "direct",
          id: params.peerId,
        },
        dmScope: params.cfg.session?.dmScope,
        identityLinks: params.cfg.session?.identityLinks,
      }),
      mainSessionKey: buildAgentMainSessionKey({
        agentId: normalizedAgentId,
        mainKey: DEFAULT_MAIN_KEY,
      }),
    };

    try {
      return await params.processMessage(params.msg, agentRoute, params.groupHistoryKey, {
        groupHistory: groupHistorySnapshot,
        suppressGroupHistoryClear: true,
      });
    } catch (err) {
      whatsappInboundLog.error(`Broadcast agent ${agentId} failed: ${formatError(err)}`);
      return false;
    }
  };

  if (strategy === "sequential") {
    for (const agentId of eligibleAgentIds) {
      await processForAgent(agentId);
    }
  } else {
    await Promise.allSettled(eligibleAgentIds.map(processForAgent));
  }

  if (params.msg.chatType === "group") {
    params.groupHistories.set(params.groupHistoryKey, []);
  }

  return true;
}

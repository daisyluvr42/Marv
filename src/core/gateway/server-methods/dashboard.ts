import { listAgentIds, resolveDefaultAgentId } from "../../../agents/agent-scope.js";
import { getKnowledgeStatusSnapshot } from "../../../knowledge/status.js";
import { getMemoryStatusSnapshot } from "../../../memory/status.js";
import { getContinuousLoopStatus, getProactiveStatusSnapshot } from "../../../proactive/status.js";
import { normalizeAgentId } from "../../../routing/session-key.js";
import { loadConfig } from "../../config/config.js";
import { ErrorCodes, errorShape } from "../protocol/index.js";
import type { GatewayRequestHandlers } from "./types.js";

function resolveDashboardAgentId(params: Record<string, unknown>) {
  const cfg = loadConfig();
  const agentIdRaw = typeof params.agentId === "string" ? params.agentId.trim() : "";
  const agentId = agentIdRaw ? normalizeAgentId(agentIdRaw) : resolveDefaultAgentId(cfg);
  if (agentIdRaw) {
    const knownAgents = listAgentIds(cfg);
    if (!knownAgents.includes(agentId)) {
      return {
        ok: false as const,
        error: errorShape(ErrorCodes.INVALID_REQUEST, `unknown agent id "${agentIdRaw}"`),
      };
    }
  }
  return {
    ok: true as const,
    cfg,
    agentId,
  };
}

export const dashboardHandlers: GatewayRequestHandlers = {
  "memory.stats": ({ params, respond }) => {
    const resolved = resolveDashboardAgentId(params);
    if (!resolved.ok) {
      respond(false, undefined, resolved.error);
      return;
    }
    respond(
      true,
      getMemoryStatusSnapshot({
        agentId: resolved.agentId,
        config: resolved.cfg,
      }),
      undefined,
    );
  },
  "knowledge.status": async ({ params, respond }) => {
    const resolved = resolveDashboardAgentId(params);
    if (!resolved.ok) {
      respond(false, undefined, resolved.error);
      return;
    }
    respond(
      true,
      await getKnowledgeStatusSnapshot({
        agentId: resolved.agentId,
        config: resolved.cfg,
      }),
      undefined,
    );
  },
  "proactive.buffer": async ({ params, respond }) => {
    const resolved = resolveDashboardAgentId(params);
    if (!resolved.ok) {
      respond(false, undefined, resolved.error);
      return;
    }
    respond(
      true,
      await getProactiveStatusSnapshot({
        agentId: resolved.agentId,
        config: resolved.cfg,
      }),
      undefined,
    );
  },
  "proactive.continuous": async ({ params, respond }) => {
    const resolved = resolveDashboardAgentId(params);
    if (!resolved.ok) {
      respond(false, undefined, resolved.error);
      return;
    }
    const budget = resolved.cfg.autonomy?.proactive?.dailyCloudTokenBudget ?? 0;
    respond(true, await getContinuousLoopStatus(resolved.agentId, budget), undefined);
  },
};

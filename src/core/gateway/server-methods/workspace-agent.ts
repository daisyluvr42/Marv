import { listAgentIds, resolveSessionAgentId } from "../../../agents/agent-scope.js";
import { normalizeAgentId } from "../../../routing/session-key.js";
import { loadConfig } from "../../config/config.js";
import { ErrorCodes, errorShape } from "../protocol/index.js";

export function resolveWorkspaceAgent(params: { agentId?: unknown; sessionKey?: unknown }):
  | {
      ok: true;
      cfg: ReturnType<typeof loadConfig>;
      agentId: string;
    }
  | {
      ok: false;
      error: ReturnType<typeof errorShape>;
    } {
  const cfg = loadConfig();
  const knownAgents = listAgentIds(cfg);
  const agentIdRaw = typeof params.agentId === "string" ? params.agentId.trim() : "";
  const sessionKey =
    typeof params.sessionKey === "string" && params.sessionKey.trim()
      ? params.sessionKey.trim()
      : undefined;
  const agentId = agentIdRaw
    ? normalizeAgentId(agentIdRaw)
    : resolveSessionAgentId({ sessionKey, config: cfg });
  if ((agentIdRaw || sessionKey) && !knownAgents.includes(agentId)) {
    return {
      ok: false,
      error: errorShape(ErrorCodes.INVALID_REQUEST, `unknown agent id "${agentIdRaw || agentId}"`),
    };
  }
  return {
    ok: true,
    cfg,
    agentId,
  };
}

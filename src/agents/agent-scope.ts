import path from "node:path";
import type { MarvConfig } from "../core/config/config.js";
import { resolveStateDir } from "../core/config/paths.js";
import type { AgentDefaultsConfig } from "../core/config/types.agent-defaults.js";
import {
  DEFAULT_AGENT_ID,
  normalizeAgentId,
  parseAgentSessionKey,
} from "../routing/session-key.js";
import { resolveUserPath } from "../utils.js";
import { normalizeSkillFilter } from "./skills/filter.js";
import { resolveDefaultAgentWorkspaceDir } from "./workspace.js";

export { resolveAgentIdFromSessionKey } from "../routing/session-key.js";

type AgentDefaults = NonNullable<NonNullable<MarvConfig["agents"]>["defaults"]>;

type ResolvedAgentConfig = {
  name?: string;
  workspace?: string;
  agentDir?: string;
  model?: AgentDefaultsConfig["model"];
  modelPool?: string;
  autoRouting?: AgentDefaults["autoRouting"];
  skills?: AgentDefaults["skills"];
  memorySearch?: AgentDefaults["memorySearch"];
  humanDelay?: AgentDefaults["humanDelay"];
  p0?: AgentDefaults["p0"];
  heartbeat?: AgentDefaults["heartbeat"];
  identity?: AgentDefaults["identity"];
  groupChat?: AgentDefaults["groupChat"];
  subagents?: AgentDefaults["subagents"];
  sandbox?: AgentDefaults["sandbox"];
  tools?: AgentDefaults["tools"];
};

function resolveDefaultsConfig(cfg: MarvConfig): AgentDefaults | undefined {
  const defaults = cfg.agents?.defaults;
  if (!defaults || typeof defaults !== "object") {
    return undefined;
  }
  return defaults;
}

export function listAgentIds(cfg: MarvConfig): string[] {
  void cfg;
  return [DEFAULT_AGENT_ID];
}

export function resolveDefaultAgentId(cfg: MarvConfig): string {
  void cfg;
  return DEFAULT_AGENT_ID;
}

export function resolveSessionAgentIds(params: { sessionKey?: string; config?: MarvConfig }): {
  defaultAgentId: string;
  sessionAgentId: string;
} {
  const defaultAgentId = resolveDefaultAgentId(params.config ?? {});
  const sessionKey = params.sessionKey?.trim();
  const normalizedSessionKey = sessionKey ? sessionKey.toLowerCase() : undefined;
  const parsed = normalizedSessionKey ? parseAgentSessionKey(normalizedSessionKey) : null;
  const sessionAgentId = parsed?.agentId ? normalizeAgentId(parsed.agentId) : defaultAgentId;
  return { defaultAgentId, sessionAgentId };
}

export function resolveSessionAgentId(params: {
  sessionKey?: string;
  config?: MarvConfig;
}): string {
  return resolveSessionAgentIds(params).sessionAgentId;
}

function resolveMainAgentConfig(cfg: MarvConfig): ResolvedAgentConfig | undefined {
  const defaults = resolveDefaultsConfig(cfg);
  if (!defaults) {
    return undefined;
  }
  return {
    name: typeof defaults.name === "string" && defaults.name.trim() ? defaults.name : undefined,
    workspace:
      typeof defaults.workspace === "string" && defaults.workspace.trim()
        ? defaults.workspace
        : undefined,
    agentDir:
      typeof defaults.agentDir === "string" && defaults.agentDir.trim()
        ? defaults.agentDir
        : undefined,
    model: defaults.model,
    modelPool:
      typeof defaults.modelPool === "string" && defaults.modelPool.trim()
        ? defaults.modelPool
        : undefined,
    autoRouting: defaults.autoRouting,
    skills: Array.isArray(defaults.skills) ? defaults.skills : undefined,
    memorySearch: defaults.memorySearch,
    humanDelay: defaults.humanDelay,
    p0: defaults.p0,
    heartbeat: defaults.heartbeat,
    identity: defaults.identity,
    groupChat: defaults.groupChat,
    subagents: defaults.subagents,
    sandbox: defaults.sandbox,
    tools: defaults.tools,
  };
}

export function resolveAgentConfig(
  cfg: MarvConfig,
  agentId: string,
): ResolvedAgentConfig | undefined {
  return normalizeAgentId(agentId) === DEFAULT_AGENT_ID ? resolveMainAgentConfig(cfg) : undefined;
}

export function resolveAgentSkillsFilter(cfg: MarvConfig, agentId: string): string[] | undefined {
  return normalizeSkillFilter(resolveAgentConfig(cfg, agentId)?.skills);
}

export function resolveAgentWorkspaceDir(cfg: MarvConfig, agentId: string) {
  const id = normalizeAgentId(agentId);
  const configured = resolveAgentConfig(cfg, id)?.workspace?.trim();
  if (configured) {
    return resolveUserPath(configured);
  }
  if (id === DEFAULT_AGENT_ID) {
    const fallback = cfg.agents?.defaults?.workspace?.trim();
    if (fallback) {
      return resolveUserPath(fallback);
    }
    return resolveDefaultAgentWorkspaceDir(process.env);
  }
  const stateDir = resolveStateDir(process.env);
  return path.join(stateDir, `workspace-${id}`);
}

export function resolveAgentDir(cfg: MarvConfig, agentId: string) {
  const id = normalizeAgentId(agentId);
  const configured = resolveAgentConfig(cfg, id)?.agentDir?.trim();
  if (configured) {
    return resolveUserPath(configured);
  }
  const root = resolveStateDir(process.env);
  return path.join(root, "agents", id, "agent");
}

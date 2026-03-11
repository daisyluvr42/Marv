import {
  resolveAgentDir,
  resolveAgentWorkspaceDir,
  resolveDefaultAgentId,
} from "../agents/agent-scope.js";
import type { AgentIdentityFile } from "../agents/prompt/identity-file.js";
import {
  identityHasValues,
  loadAgentIdentityFromWorkspace,
  parseIdentityMarkdown as parseIdentityMarkdownFile,
} from "../agents/prompt/identity-file.js";
import type { MarvConfig } from "../core/config/config.js";
import { normalizeAgentId } from "../routing/session-key.js";

export type AgentSummary = {
  id: string;
  name?: string;
  identityName?: string;
  identityEmoji?: string;
  identitySource?: "identity" | "config";
  workspace: string;
  agentDir: string;
  model?: string;
  providers?: string[];
  isDefault: boolean;
};

export type AgentIdentity = AgentIdentityFile;

function resolveAgentName(cfg: MarvConfig, agentId: string) {
  const normalizedId = normalizeAgentId(agentId);
  if (normalizedId !== normalizeAgentId(resolveDefaultAgentId(cfg))) {
    return undefined;
  }
  const defaultName = cfg.agents?.defaults?.name?.trim();
  return defaultName || undefined;
}

function resolveAgentModel(cfg: MarvConfig, agentId: string) {
  const normalizedId = normalizeAgentId(agentId);
  if (normalizedId !== normalizeAgentId(resolveDefaultAgentId(cfg))) {
    return undefined;
  }
  const raw = cfg.agents?.defaults?.model;
  if (typeof raw === "string") {
    return raw;
  }
  return raw?.primary?.trim() || undefined;
}

export function parseIdentityMarkdown(content: string): AgentIdentity {
  return parseIdentityMarkdownFile(content);
}

export function loadAgentIdentity(workspace: string): AgentIdentity | null {
  const parsed = loadAgentIdentityFromWorkspace(workspace);
  if (!parsed) {
    return null;
  }
  return identityHasValues(parsed) ? parsed : null;
}

export function buildAgentSummaries(cfg: MarvConfig): AgentSummary[] {
  const defaultAgentId = normalizeAgentId(resolveDefaultAgentId(cfg));
  const workspace = resolveAgentWorkspaceDir(cfg, defaultAgentId);
  const identity = loadAgentIdentity(workspace);
  const configIdentity = cfg.agents?.defaults?.identity;
  const identityName = identity?.name ?? configIdentity?.name?.trim();
  const identityEmoji = identity?.emoji ?? configIdentity?.emoji?.trim();
  const identitySource = identity
    ? "identity"
    : configIdentity && (identityName || identityEmoji)
      ? "config"
      : undefined;

  return [
    {
      id: defaultAgentId,
      name: resolveAgentName(cfg, defaultAgentId),
      identityName,
      identityEmoji,
      identitySource,
      workspace,
      agentDir: resolveAgentDir(cfg, defaultAgentId),
      model: resolveAgentModel(cfg, defaultAgentId),
      isDefault: true,
    },
  ];
}

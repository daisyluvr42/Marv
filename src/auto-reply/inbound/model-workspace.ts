import {
  resolveAgentDir,
  resolveAgentWorkspaceDir,
  resolveSessionAgentId,
  resolveAgentSkillsFilter,
} from "../../agents/agent-scope.js";
import { resolveModelRefFromString } from "../../agents/model/model-selection.js";
import { resolveAgentTimeoutMs } from "../../agents/timeout.js";
import { DEFAULT_AGENT_WORKSPACE_DIR, ensureAgentWorkspace } from "../../agents/workspace.js";
import { type MarvConfig, loadConfig } from "../../core/config/config.js";
import { defaultRuntime } from "../../runtime.js";
import { createTypingController, type TypingController } from "../delivery/typing.js";
import { resolveDefaultModel } from "../directives/index.js";
import type { TurnContext } from "../support/templating.js";
import { SILENT_REPLY_TOKEN } from "../support/tokens.js";
import { isHeartbeatRun } from "../support/types.js";
import type { GetReplyOptions } from "../support/types.js";

function mergeSkillFilters(channelFilter?: string[], agentFilter?: string[]): string[] | undefined {
  const normalize = (list?: string[]) => {
    if (!Array.isArray(list)) {
      return undefined;
    }
    return list.map((entry) => String(entry).trim()).filter(Boolean);
  };
  const channel = normalize(channelFilter);
  const agent = normalize(agentFilter);
  if (!channel && !agent) {
    return undefined;
  }
  if (!channel) {
    return agent;
  }
  if (!agent) {
    return channel;
  }
  if (channel.length === 0 || agent.length === 0) {
    return [];
  }
  const agentSet = new Set(agent);
  return channel.filter((name) => agentSet.has(name));
}

export type ModelWorkspaceResult = {
  cfg: MarvConfig;
  agentId: string;
  agentDir: string;
  agentCfg: NonNullable<MarvConfig["agents"]>["defaults"];
  sessionCfg: MarvConfig["session"];
  workspaceDir: string;
  provider: string;
  model: string;
  defaultProvider: string;
  defaultModel: string;
  aliasIndex: ReturnType<typeof resolveDefaultModel>["aliasIndex"];
  hasResolvedHeartbeatModelOverride: boolean;
  autoRoutingThinking: string | undefined;
  timeoutMs: number;
  typing: TypingController;
  mergedSkillFilter: string[] | undefined;
  resolvedOpts: GetReplyOptions | undefined;
  isFastTestEnv: boolean;
};

/**
 * Stage 1: Resolve model, workspace, agent config, and typing controller.
 * Pure setup — no side effects on the inbound context.
 */
export async function resolveModelAndWorkspace(
  ctx: TurnContext,
  opts: GetReplyOptions | undefined,
  configOverride: MarvConfig | undefined,
): Promise<ModelWorkspaceResult> {
  const isFastTestEnv = process.env.MARV_TEST_FAST === "1";
  const cfg = configOverride ?? loadConfig();
  const targetSessionKey =
    ctx.CommandSource === "native" ? ctx.CommandTargetSessionKey?.trim() : undefined;
  const agentSessionKey = targetSessionKey || ctx.SessionKey;
  const agentId = resolveSessionAgentId({
    sessionKey: agentSessionKey,
    config: cfg,
  });
  const mergedSkillFilter = mergeSkillFilters(
    opts?.skillFilter,
    resolveAgentSkillsFilter(cfg, agentId),
  );
  const resolvedOpts =
    mergedSkillFilter !== undefined ? { ...opts, skillFilter: mergedSkillFilter } : opts;
  const agentCfg = cfg.agents?.defaults;
  const sessionCfg = cfg.session;
  const { defaultProvider, defaultModel, aliasIndex } = resolveDefaultModel({
    cfg,
    agentId,
  });
  let provider = defaultProvider;
  let model = defaultModel;
  let hasResolvedHeartbeatModelOverride = false;
  if (isHeartbeatRun(opts)) {
    const heartbeatRaw =
      opts?.heartbeatModelOverride?.trim() ?? agentCfg?.heartbeat?.model?.trim() ?? "";
    const heartbeatRef = heartbeatRaw
      ? resolveModelRefFromString({
          raw: heartbeatRaw,
          defaultProvider,
          aliasIndex,
        })
      : null;
    if (heartbeatRef) {
      provider = heartbeatRef.ref.provider;
      model = heartbeatRef.ref.model;
      hasResolvedHeartbeatModelOverride = true;
    }
  }

  // Auto-routing is scoped to subagent model selection only (sessions_spawn).
  // The main session model is chosen manually by the user; auto-routing does
  // not override it or its thinking level.
  const autoRoutingThinking: string | undefined = undefined;

  const workspaceDirRaw = resolveAgentWorkspaceDir(cfg, agentId) ?? DEFAULT_AGENT_WORKSPACE_DIR;
  const workspace = await ensureAgentWorkspace({
    dir: workspaceDirRaw,
    ensureBootstrapFiles: !agentCfg?.skipBootstrap && !isFastTestEnv,
  });
  const workspaceDir = workspace.dir;
  const agentDir = resolveAgentDir(cfg, agentId);
  const timeoutMs = resolveAgentTimeoutMs({ cfg, overrideSeconds: opts?.timeoutOverrideSeconds });
  const configuredTypingSeconds =
    agentCfg?.typingIntervalSeconds ?? sessionCfg?.typingIntervalSeconds;
  const typingIntervalSeconds =
    typeof configuredTypingSeconds === "number" ? configuredTypingSeconds : 6;
  const typing = createTypingController({
    onReplyStart: opts?.onReplyStart,
    onCleanup: opts?.onTypingCleanup,
    typingIntervalSeconds,
    silentToken: SILENT_REPLY_TOKEN,
    log: defaultRuntime.log,
  });
  opts?.onTypingController?.(typing);

  return {
    cfg,
    agentId,
    agentDir,
    agentCfg,
    sessionCfg,
    workspaceDir,
    provider,
    model,
    defaultProvider,
    defaultModel,
    aliasIndex,
    hasResolvedHeartbeatModelOverride,
    autoRoutingThinking,
    timeoutMs,
    typing,
    mergedSkillFilter,
    resolvedOpts,
    isFastTestEnv,
  };
}

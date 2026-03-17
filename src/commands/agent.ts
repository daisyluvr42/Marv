import {
  listAgentIds,
  resolveAgentDir,
  resolveAgentSkillsFilter,
  resolveAgentWorkspaceDir,
} from "../agents/agent-scope.js";
import { ensureAuthProfileStore } from "../agents/auth-profiles.js";
import { clearSessionAuthProfileOverride } from "../agents/auth-profiles/session-override.js";
import { resolveAutoRouting } from "../agents/auto-routing.js";
import { getCliSessionId } from "../agents/cli-session.js";
import { DEFAULT_MODEL, DEFAULT_PROVIDER } from "../agents/defaults.js";
import { AGENT_LANE_SUBAGENT } from "../agents/lanes.js";
import { loadModelCatalog } from "../agents/model/model-catalog.js";
import { runWithModelFallback } from "../agents/model/model-fallback.js";
import { resolveRuntimeModelPlan } from "../agents/model/model-pool.js";
import {
  isCliProvider,
  modelKey,
  normalizeModelRef,
  resolveThinkingDefault,
} from "../agents/model/model-selection.js";
import { runCliAgent } from "../agents/runner/cli-runner.js";
import { runEmbeddedPiAgent } from "../agents/runner/pi-embedded.js";
import { buildWorkspaceSkillSnapshot } from "../agents/skills.js";
import { getSkillsSnapshotVersion } from "../agents/skills/refresh.js";
import { resolveAgentTimeoutMs } from "../agents/timeout.js";
import { ensureAgentWorkspace } from "../agents/workspace.js";
import {
  formatThinkingLevels,
  formatXHighModelHint,
  normalizeThinkLevel,
  normalizeVerboseLevel,
  supportsXHighThinking,
  type ThinkLevel,
  type VerboseLevel,
} from "../auto-reply/thinking.js";
import { formatCliCommand } from "../cli/command-format.js";
import { type CliDeps, createDefaultDeps } from "../cli/deps.js";
import { loadConfig } from "../core/config/config.js";
import {
  resolveAgentIdFromSessionKey,
  resolveSessionFilePath,
  type SessionEntry,
  updateSessionStore,
} from "../core/config/sessions.js";
import { applyVerboseOverride } from "../core/session/level-overrides.js";
import { applyModelOverrideToSessionEntry } from "../core/session/model-overrides.js";
import { resolveSessionModelSelectionState } from "../core/session/model-selection-state.js";
import { resolveSendPolicy } from "../core/session/send-policy.js";
import {
  clearAgentRunContext,
  emitAgentEvent,
  registerAgentRunContext,
} from "../infra/agent-events.js";
import { getRemoteSkillEligibility } from "../infra/skills-remote.js";
import { normalizeAgentId } from "../routing/session-key.js";
import { defaultRuntime, type RuntimeEnv } from "../runtime.js";
import { resolveMessageChannel } from "../utils/message-channel.js";
import { deliverAgentCommandResult } from "./agent/delivery.js";
import { resolveAgentRunContext } from "./agent/run-context.js";
import { updateSessionStoreAfterAgentRun } from "./agent/session-store.js";
import { resolveSession } from "./agent/session.js";
import type { AgentCommandOpts } from "./agent/types.js";

type PersistSessionEntryParams = {
  sessionStore: Record<string, SessionEntry>;
  sessionKey: string;
  storePath: string;
  entry: SessionEntry;
};

async function persistSessionEntry(params: PersistSessionEntryParams): Promise<void> {
  params.sessionStore[params.sessionKey] = params.entry;
  await updateSessionStore(params.storePath, (store) => {
    store[params.sessionKey] = params.entry;
  });
}

function resolveFallbackRetryPrompt(params: { body: string; isFallbackRetry: boolean }): string {
  if (!params.isFallbackRetry) {
    return params.body;
  }
  return "Continue where you left off. The previous model attempt failed or timed out.";
}

function runAgentAttempt(params: {
  providerOverride: string;
  modelOverride: string;
  cfg: ReturnType<typeof loadConfig>;
  sessionEntry: SessionEntry | undefined;
  sessionId: string;
  sessionKey: string | undefined;
  sessionAgentId: string;
  sessionFile: string;
  workspaceDir: string;
  body: string;
  isFallbackRetry: boolean;
  resolvedThinkLevel: ThinkLevel;
  timeoutMs: number;
  runId: string;
  opts: AgentCommandOpts;
  runContext: ReturnType<typeof resolveAgentRunContext>;
  spawnedBy: string | undefined;
  messageChannel: ReturnType<typeof resolveMessageChannel>;
  skillsSnapshot: ReturnType<typeof buildWorkspaceSkillSnapshot> | undefined;
  resolvedVerboseLevel: VerboseLevel | undefined;
  agentDir: string;
  onAgentEvent: (evt: { stream: string; data?: Record<string, unknown> }) => void;
  primaryProvider: string;
}) {
  const effectivePrompt = resolveFallbackRetryPrompt({
    body: params.body,
    isFallbackRetry: params.isFallbackRetry,
  });
  if (isCliProvider(params.providerOverride, params.cfg)) {
    const cliSessionId = getCliSessionId(params.sessionEntry, params.providerOverride);
    return runCliAgent({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      agentId: params.sessionAgentId,
      sessionFile: params.sessionFile,
      workspaceDir: params.workspaceDir,
      config: params.cfg,
      prompt: effectivePrompt,
      provider: params.providerOverride,
      model: params.modelOverride,
      thinkLevel: params.resolvedThinkLevel,
      timeoutMs: params.timeoutMs,
      runId: params.runId,
      extraSystemPrompt: params.opts.extraSystemPrompt,
      cliSessionId,
      images: params.isFallbackRetry ? undefined : params.opts.images,
      streamParams: params.opts.streamParams,
    });
  }

  const authProfileId =
    params.providerOverride === params.primaryProvider
      ? params.sessionEntry?.authProfileOverride
      : undefined;
  return runEmbeddedPiAgent({
    sessionId: params.sessionId,
    sessionKey: params.sessionKey,
    agentId: params.sessionAgentId,
    messageChannel: params.messageChannel,
    agentAccountId: params.runContext.accountId,
    messageTo: params.opts.replyTo ?? params.opts.to,
    messageThreadId: params.opts.threadId,
    groupId: params.runContext.groupId,
    groupChannel: params.runContext.groupChannel,
    groupSpace: params.runContext.groupSpace,
    spawnedBy: params.spawnedBy,
    currentChannelId: params.runContext.currentChannelId,
    currentThreadTs: params.runContext.currentThreadTs,
    replyToMode: params.runContext.replyToMode,
    hasRepliedRef: params.runContext.hasRepliedRef,
    senderIsOwner: true,
    sessionFile: params.sessionFile,
    workspaceDir: params.workspaceDir,
    config: params.cfg,
    skillsSnapshot: params.skillsSnapshot,
    prompt: effectivePrompt,
    images: params.isFallbackRetry ? undefined : params.opts.images,
    clientTools: params.opts.clientTools,
    provider: params.providerOverride,
    model: params.modelOverride,
    authProfileId,
    authProfileIdSource: authProfileId ? params.sessionEntry?.authProfileOverrideSource : undefined,
    thinkLevel: params.resolvedThinkLevel,
    verboseLevel: params.resolvedVerboseLevel,
    timeoutMs: params.timeoutMs,
    runId: params.runId,
    lane: params.opts.lane,
    abortSignal: params.opts.abortSignal,
    extraSystemPrompt: params.opts.extraSystemPrompt,
    inputProvenance: params.opts.inputProvenance,
    streamParams: params.opts.streamParams,
    agentDir: params.agentDir,
    onAgentEvent: params.onAgentEvent,
  });
}

export async function agentCommand(
  opts: AgentCommandOpts,
  runtime: RuntimeEnv = defaultRuntime,
  deps: CliDeps = createDefaultDeps(),
) {
  const body = (opts.message ?? "").trim();
  if (!body) {
    throw new Error("Message (--message) is required");
  }
  if (!opts.to && !opts.sessionId && !opts.sessionKey && !opts.agentId) {
    throw new Error("Pass --to <E.164>, --session-id, or --agent to choose a session");
  }

  const cfg = loadConfig();
  const agentIdOverrideRaw = opts.agentId?.trim();
  const agentIdOverride = agentIdOverrideRaw ? normalizeAgentId(agentIdOverrideRaw) : undefined;
  if (agentIdOverride) {
    const knownAgents = listAgentIds(cfg);
    if (!knownAgents.includes(agentIdOverride)) {
      throw new Error(
        `Unknown agent id "${agentIdOverrideRaw}". Use "${formatCliCommand("marv agents list")}" to see configured agents.`,
      );
    }
  }
  if (agentIdOverride && opts.sessionKey) {
    const sessionAgentId = resolveAgentIdFromSessionKey(opts.sessionKey);
    if (sessionAgentId !== agentIdOverride) {
      throw new Error(
        `Agent id "${agentIdOverrideRaw}" does not match session key agent "${sessionAgentId}".`,
      );
    }
  }
  const agentCfg = cfg.agents?.defaults;
  const sessionAgentId = agentIdOverride ?? resolveAgentIdFromSessionKey(opts.sessionKey?.trim());
  const workspaceDirRaw = resolveAgentWorkspaceDir(cfg, sessionAgentId);
  const agentDir = resolveAgentDir(cfg, sessionAgentId);
  const workspace = await ensureAgentWorkspace({
    dir: workspaceDirRaw,
    ensureBootstrapFiles: !agentCfg?.skipBootstrap,
  });
  const workspaceDir = workspace.dir;
  const configuredPlan = resolveRuntimeModelPlan({ cfg, agentId: sessionAgentId, agentDir });
  const configuredModel = configuredPlan.candidates[0] ?? {
    provider: DEFAULT_PROVIDER,
    model: DEFAULT_MODEL,
  };
  const thinkingLevelsHint = formatThinkingLevels(configuredModel.provider, configuredModel.model);

  const thinkOverride = normalizeThinkLevel(opts.thinking);
  const thinkOnce = normalizeThinkLevel(opts.thinkingOnce);
  if (opts.thinking && !thinkOverride) {
    throw new Error(`Invalid thinking level. Use one of: ${thinkingLevelsHint}.`);
  }
  if (opts.thinkingOnce && !thinkOnce) {
    throw new Error(`Invalid one-shot thinking level. Use one of: ${thinkingLevelsHint}.`);
  }

  const verboseOverride = normalizeVerboseLevel(opts.verbose);
  if (opts.verbose && !verboseOverride) {
    throw new Error('Invalid verbose level. Use "on", "full", or "off".');
  }

  const laneRaw = typeof opts.lane === "string" ? opts.lane.trim() : "";
  const isSubagentLane = laneRaw === String(AGENT_LANE_SUBAGENT);
  const timeoutSecondsRaw =
    opts.timeout !== undefined
      ? Number.parseInt(String(opts.timeout), 10)
      : isSubagentLane
        ? 0
        : undefined;
  if (
    timeoutSecondsRaw !== undefined &&
    (Number.isNaN(timeoutSecondsRaw) || timeoutSecondsRaw < 0)
  ) {
    throw new Error("--timeout must be a non-negative integer (seconds; 0 means no timeout)");
  }
  const timeoutMs = resolveAgentTimeoutMs({
    cfg,
    overrideSeconds: timeoutSecondsRaw,
  });

  const sessionResolution = resolveSession({
    cfg,
    to: opts.to,
    sessionId: opts.sessionId,
    sessionKey: opts.sessionKey,
    agentId: agentIdOverride,
  });

  const {
    sessionId,
    sessionKey,
    sessionEntry: resolvedSessionEntry,
    sessionStore,
    storePath,
    isNewSession,
    persistedThinking,
    persistedVerbose,
  } = sessionResolution;
  let sessionEntry = resolvedSessionEntry;
  const runId = opts.runId?.trim() || sessionId;

  try {
    if (opts.deliver === true) {
      const sendPolicy = resolveSendPolicy({
        cfg,
        entry: sessionEntry,
        sessionKey,
        channel: sessionEntry?.channel,
        chatType: sessionEntry?.chatType,
      });
      if (sendPolicy === "deny") {
        throw new Error("send blocked by session policy");
      }
    }

    let resolvedThinkLevel =
      thinkOnce ??
      thinkOverride ??
      persistedThinking ??
      (agentCfg?.thinkingDefault as ThinkLevel | undefined);
    const resolvedVerboseLevel =
      verboseOverride ?? persistedVerbose ?? (agentCfg?.verboseDefault as VerboseLevel | undefined);

    if (sessionKey) {
      registerAgentRunContext(runId, {
        sessionKey,
        verboseLevel: resolvedVerboseLevel,
        runModeKind: "user",
      });
    }

    const needsSkillsSnapshot = isNewSession || !sessionEntry?.skillsSnapshot;
    const skillsSnapshotVersion = getSkillsSnapshotVersion(workspaceDir);
    const skillFilter = resolveAgentSkillsFilter(cfg, sessionAgentId);
    const skillsSnapshot = needsSkillsSnapshot
      ? buildWorkspaceSkillSnapshot(workspaceDir, {
          config: cfg,
          eligibility: { remote: getRemoteSkillEligibility() },
          snapshotVersion: skillsSnapshotVersion,
          skillFilter,
        })
      : sessionEntry?.skillsSnapshot;

    if (skillsSnapshot && sessionStore && sessionKey && needsSkillsSnapshot) {
      const current = sessionEntry ?? {
        sessionId,
        updatedAt: Date.now(),
      };
      const next: SessionEntry = {
        ...current,
        sessionId,
        updatedAt: Date.now(),
        skillsSnapshot,
      };
      await persistSessionEntry({
        sessionStore,
        sessionKey,
        storePath,
        entry: next,
      });
      sessionEntry = next;
    }

    // Persist explicit /command overrides to the session store when we have a key.
    if (sessionStore && sessionKey) {
      const entry = sessionStore[sessionKey] ??
        sessionEntry ?? { sessionId, updatedAt: Date.now() };
      const next: SessionEntry = { ...entry, sessionId, updatedAt: Date.now() };
      if (thinkOverride) {
        next.thinkingLevel = thinkOverride;
      }
      applyVerboseOverride(next, verboseOverride);
      await persistSessionEntry({
        sessionStore,
        sessionKey,
        storePath,
        entry: next,
      });
    }

    const runtimePlan = resolveRuntimeModelPlan({
      cfg,
      agentId: sessionAgentId,
      agentDir,
      requirements: {
        requiredCapabilities: (opts.images?.length ?? 0) > 0 ? ["text", "vision"] : ["text"],
      },
    });
    const configuredDefaultRef = runtimePlan.candidates[0] ?? {
      provider: DEFAULT_PROVIDER,
      model: DEFAULT_MODEL,
    };
    const { provider: defaultProvider, model: defaultModel } = normalizeModelRef(
      configuredDefaultRef.provider,
      configuredDefaultRef.model,
    );
    let provider = defaultProvider;
    let model = defaultModel;
    const hasStoredOverride = Boolean(
      sessionEntry?.modelOverride || sessionEntry?.providerOverride,
    );
    let allowedModelKeys = new Set<string>(
      runtimePlan.candidates.map((entry) => modelKey(entry.provider, entry.model)),
    );
    let modelCatalog: Awaited<ReturnType<typeof loadModelCatalog>> | null = null;

    if (sessionEntry && sessionStore && sessionKey && hasStoredOverride) {
      const entry = sessionEntry;
      const selectionState = resolveSessionModelSelectionState(entry);
      const overrideRef =
        selectionState.mode === "manual" ? selectionState.manualModelRef : undefined;
      if (overrideRef) {
        const slash = overrideRef.indexOf("/");
        const overrideProvider =
          slash > 0
            ? overrideRef.slice(0, slash)
            : sessionEntry.providerOverride?.trim() || defaultProvider;
        const overrideModel = slash > 0 ? overrideRef.slice(slash + 1) : overrideRef;
        const normalizedOverride = normalizeModelRef(overrideProvider, overrideModel);
        const key = modelKey(normalizedOverride.provider, normalizedOverride.model);
        if (
          !isCliProvider(normalizedOverride.provider, cfg) &&
          allowedModelKeys.size > 0 &&
          !allowedModelKeys.has(key)
        ) {
          const { updated } = applyModelOverrideToSessionEntry({
            entry,
            selection: { provider: defaultProvider, model: defaultModel, isDefault: true },
          });
          if (updated) {
            await persistSessionEntry({
              sessionStore,
              sessionKey,
              storePath,
              entry,
            });
          }
        }
      }
    }

    const selectionState = resolveSessionModelSelectionState(sessionEntry);
    const storedModelOverride =
      selectionState.mode === "manual" ? selectionState.manualModelRef : undefined;
    if (storedModelOverride) {
      const slash = storedModelOverride.indexOf("/");
      const candidateProvider =
        slash > 0
          ? storedModelOverride.slice(0, slash)
          : sessionEntry?.providerOverride?.trim() || defaultProvider;
      const candidateModel = slash > 0 ? storedModelOverride.slice(slash + 1) : storedModelOverride;
      const normalizedStored = normalizeModelRef(candidateProvider, candidateModel);
      const key = modelKey(normalizedStored.provider, normalizedStored.model);
      if (
        isCliProvider(normalizedStored.provider, cfg) ||
        allowedModelKeys.size === 0 ||
        allowedModelKeys.has(key)
      ) {
        provider = normalizedStored.provider;
        model = normalizedStored.model;
      }
    }
    if (sessionEntry) {
      const authProfileId = sessionEntry.authProfileOverride;
      if (authProfileId) {
        const entry = sessionEntry;
        const store = ensureAuthProfileStore();
        const profile = store.profiles[authProfileId];
        if (!profile || profile.provider !== provider) {
          if (sessionStore && sessionKey) {
            await clearSessionAuthProfileOverride({
              sessionEntry: entry,
              sessionStore,
              sessionKey,
              storePath,
            });
          }
        }
      }
    }

    if (!resolvedThinkLevel) {
      let catalogForThinking: Awaited<ReturnType<typeof loadModelCatalog>> = modelCatalog ?? [];
      if (catalogForThinking.length === 0) {
        modelCatalog = await loadModelCatalog({ config: cfg });
        catalogForThinking = modelCatalog;
      }
      resolvedThinkLevel = resolveThinkingDefault({
        cfg,
        provider,
        model,
        catalog: catalogForThinking,
      });
    }
    if (resolvedThinkLevel === "xhigh" && !supportsXHighThinking(provider, model)) {
      const explicitThink = Boolean(thinkOnce || thinkOverride);
      if (explicitThink) {
        throw new Error(`Thinking level "xhigh" is only supported for ${formatXHighModelHint()}.`);
      }
      resolvedThinkLevel = "high";
      if (sessionEntry && sessionStore && sessionKey && sessionEntry.thinkingLevel === "xhigh") {
        const entry = sessionEntry;
        entry.thinkingLevel = "high";
        entry.updatedAt = Date.now();
        await persistSessionEntry({
          sessionStore,
          sessionKey,
          storePath,
          entry,
        });
      }
    }
    const sessionFile = resolveSessionFilePath(sessionId, sessionEntry, {
      agentId: sessionAgentId,
    });

    const startedAt = Date.now();
    let lifecycleEnded = false;

    let result: Awaited<ReturnType<typeof runEmbeddedPiAgent>>;
    let fallbackProvider = provider;
    let fallbackModel = model;
    try {
      const runContext = resolveAgentRunContext(opts);
      const messageChannel = resolveMessageChannel(
        runContext.messageChannel,
        opts.replyChannel ?? opts.channel,
      );
      const spawnedBy = opts.spawnedBy ?? sessionEntry?.spawnedBy;
      const activeFallbacks = runtimePlan.candidates
        .map((entry) => entry.ref)
        .filter((ref) => ref !== `${provider}/${model}`);
      let routedThinking: ThinkLevel | undefined = undefined;

      // Auto-routing may still tune policy hints like thinking, but no longer picks models.
      const autoRoutingResult = await resolveAutoRouting({
        prompt: body,
        hasImages: (opts.images?.length ?? 0) > 0,
        config: cfg,
        agentId: sessionAgentId,
        defaultProvider: provider,
        defaultModel: model,
      });

      if (autoRoutingResult.routed) {
        if (autoRoutingResult.thinking) {
          routedThinking = autoRoutingResult.thinking;
        }
      }

      // Track model fallback attempts so retries on an existing session don't
      // re-inject the original prompt as a duplicate user message.
      let fallbackAttemptIndex = 0;
      const fallbackResult = await runWithModelFallback({
        cfg,
        provider,
        model,
        agentDir,
        fallbacksOverride: activeFallbacks,
        run: (providerOverride, modelOverride) => {
          const isFallbackRetry = fallbackAttemptIndex > 0;
          fallbackAttemptIndex += 1;
          return runAgentAttempt({
            providerOverride,
            modelOverride,
            cfg,
            sessionEntry,
            sessionId,
            sessionKey,
            sessionAgentId,
            sessionFile,
            workspaceDir,
            body,
            isFallbackRetry,
            resolvedThinkLevel: routedThinking ?? resolvedThinkLevel,
            timeoutMs,
            runId,
            opts,
            runContext,
            spawnedBy,
            messageChannel,
            skillsSnapshot,
            resolvedVerboseLevel,
            agentDir,
            primaryProvider: provider,
            onAgentEvent: (evt) => {
              // Track lifecycle end for fallback emission below.
              if (
                evt.stream === "lifecycle" &&
                typeof evt.data?.phase === "string" &&
                (evt.data.phase === "end" || evt.data.phase === "error")
              ) {
                lifecycleEnded = true;
              }
            },
          });
        },
      });
      result = fallbackResult.result;
      fallbackProvider = fallbackResult.provider;
      fallbackModel = fallbackResult.model;
      if (!lifecycleEnded) {
        emitAgentEvent({
          runId,
          stream: "lifecycle",
          data: {
            phase: "end",
            startedAt,
            endedAt: Date.now(),
            aborted: result.meta.aborted ?? false,
          },
        });
      }
    } catch (err) {
      if (!lifecycleEnded) {
        emitAgentEvent({
          runId,
          stream: "lifecycle",
          data: {
            phase: "error",
            startedAt,
            endedAt: Date.now(),
            error: String(err),
          },
        });
      }
      throw err;
    }

    // Update token+model fields in the session store.
    if (sessionStore && sessionKey) {
      await updateSessionStoreAfterAgentRun({
        cfg,
        contextTokensOverride: agentCfg?.contextTokens,
        sessionId,
        sessionKey,
        storePath,
        sessionStore,
        defaultProvider: provider,
        defaultModel: model,
        fallbackProvider,
        fallbackModel,
        result,
      });
    }

    const payloads = result.payloads ?? [];
    return await deliverAgentCommandResult({
      cfg,
      deps,
      runtime,
      opts,
      sessionEntry,
      result,
      payloads,
    });
  } finally {
    clearAgentRunContext(runId);
  }
}

import type { MarvConfig } from "../core/config/config.js";
import { resolveReplyDirectives } from "./directives/get-reply-directives.js";
import { runPreparedReply } from "./execution/get-reply-run.js";
import { prepareExecution } from "./execution/prepare.js";
import { enrichInbound } from "./inbound/enrich.js";
import { resolveModelAndWorkspace } from "./inbound/model-workspace.js";
import { handleInlineActions } from "./pipeline-inline-actions.js";
import { initSessionState } from "./session/init.js";
import { applyResetModelOverride } from "./session/reset-model.js";
import type { TurnContext } from "./support/templating.js";
import type { GetReplyOptions, ReplyPayload } from "./support/types.js";

/**
 * Main turn pipeline: transforms an inbound message into an agent reply.
 *
 * Each numbered stage is a self-contained function; this coordinator
 * wires them together and handles early-return branches.
 */
export async function getReplyFromConfig(
  ctx: TurnContext,
  opts?: GetReplyOptions,
  configOverride?: MarvConfig,
): Promise<ReplyPayload | ReplyPayload[] | undefined> {
  // Stage 1 — resolve model, workspace, agent config, typing controller
  const mw = await resolveModelAndWorkspace(ctx, opts, configOverride);
  let { provider, model } = mw;

  // Stage 2 — finalize inbound context, run media/link understanding, resolve command auth
  const { finalized, commandAuthorized } = await enrichInbound({
    ctx,
    cfg: mw.cfg,
    agentDir: mw.agentDir,
    provider,
    model,
    isFastTestEnv: mw.isFastTestEnv,
  });

  // Stage 3 — initialize session state and apply reset-model override
  const sessionState = await initSessionState({
    ctx: finalized,
    cfg: mw.cfg,
    commandAuthorized,
  });
  let { abortedLastRun } = sessionState;
  const {
    sessionCtx,
    sessionEntry,
    previousSessionEntry,
    sessionStore,
    sessionKey,
    sessionId,
    isNewSession,
    resetTriggered,
    systemSent,
    storePath,
    sessionScope,
    groupResolution,
    isGroup,
    triggerBodyNormalized,
    bodyStripped,
  } = sessionState;

  await applyResetModelOverride({
    cfg: mw.cfg,
    resetTriggered,
    bodyStripped,
    sessionCtx,
    ctx: finalized,
    sessionEntry,
    sessionStore,
    sessionKey,
    storePath,
    defaultProvider: mw.defaultProvider,
    defaultModel: mw.defaultModel,
    aliasIndex: mw.aliasIndex,
  });

  // Stage 4 — resolve inline directives (model, thinking, elevated, etc.)
  const directiveResult = await resolveReplyDirectives({
    ctx: finalized,
    cfg: mw.cfg,
    agentId: mw.agentId,
    agentDir: mw.agentDir,
    workspaceDir: mw.workspaceDir,
    agentCfg: mw.agentCfg,
    sessionCtx,
    sessionEntry,
    sessionStore,
    sessionKey,
    storePath,
    sessionScope,
    groupResolution,
    isGroup,
    triggerBodyNormalized,
    commandAuthorized,
    defaultProvider: mw.defaultProvider,
    defaultModel: mw.defaultModel,
    aliasIndex: mw.aliasIndex,
    provider,
    model,
    hasResolvedHeartbeatModelOverride: mw.hasResolvedHeartbeatModelOverride,
    autoRoutingThinking: mw.autoRoutingThinking,
    typing: mw.typing,
    opts: mw.resolvedOpts,
    skillFilter: mw.mergedSkillFilter,
  });
  if (directiveResult.kind === "reply") {
    return directiveResult.reply;
  }
  const dr = directiveResult.result;
  provider = dr.provider;
  model = dr.model;

  // Stage 5 — handle inline actions (commands, skill dispatch, status)
  const inlineActionResult = await handleInlineActions({
    ctx,
    sessionCtx,
    cfg: mw.cfg,
    agentId: mw.agentId,
    agentDir: mw.agentDir,
    sessionEntry,
    previousSessionEntry,
    sessionStore,
    sessionKey,
    storePath,
    sessionScope,
    workspaceDir: mw.workspaceDir,
    isGroup,
    opts: mw.resolvedOpts,
    typing: mw.typing,
    allowTextCommands: dr.allowTextCommands,
    inlineStatusRequested: dr.inlineStatusRequested,
    command: dr.command,
    skillCommands: dr.skillCommands,
    directives: dr.directives,
    cleanedBody: dr.cleanedBody,
    elevatedEnabled: dr.elevatedEnabled,
    elevatedAllowed: dr.elevatedAllowed,
    elevatedFailures: dr.elevatedFailures,
    defaultActivation: () => dr.defaultActivation,
    resolvedThinkLevel: dr.resolvedThinkLevel,
    resolvedVerboseLevel: dr.resolvedVerboseLevel,
    resolvedReasoningLevel: dr.resolvedReasoningLevel,
    resolvedElevatedLevel: dr.resolvedElevatedLevel,
    resolveDefaultThinkingLevel: dr.modelState.resolveDefaultThinkingLevel,
    provider,
    model,
    contextTokens: dr.contextTokens,
    directiveAck: dr.directiveAck,
    abortedLastRun,
    skillFilter: mw.mergedSkillFilter,
  });
  if (inlineActionResult.kind === "reply") {
    return inlineActionResult.reply;
  }
  const directives = inlineActionResult.directives;
  abortedLastRun = inlineActionResult.abortedLastRun ?? abortedLastRun;

  // Stage 6 — stage sandbox media and load prompt images
  const { runReplyOpts } = await prepareExecution({
    ctx,
    sessionCtx,
    cfg: mw.cfg,
    sessionKey,
    workspaceDir: mw.workspaceDir,
    resolvedOpts: mw.resolvedOpts,
  });

  // Stage 7 — run the agent and produce the reply
  return runPreparedReply({
    ctx,
    sessionCtx,
    cfg: mw.cfg,
    agentId: mw.agentId,
    agentDir: mw.agentDir,
    agentCfg: mw.agentCfg,
    sessionCfg: mw.sessionCfg,
    commandAuthorized,
    command: dr.command,
    commandSource: dr.commandSource,
    allowTextCommands: dr.allowTextCommands,
    directives,
    defaultActivation: dr.defaultActivation,
    resolvedThinkLevel: dr.resolvedThinkLevel,
    resolvedVerboseLevel: dr.resolvedVerboseLevel,
    resolvedReasoningLevel: dr.resolvedReasoningLevel,
    resolvedElevatedLevel: dr.resolvedElevatedLevel,
    execOverrides: dr.execOverrides,
    elevatedEnabled: dr.elevatedEnabled,
    elevatedAllowed: dr.elevatedAllowed,
    blockStreamingEnabled: dr.blockStreamingEnabled,
    blockReplyChunking: dr.blockReplyChunking,
    resolvedBlockStreamingBreak: dr.resolvedBlockStreamingBreak,
    modelState: dr.modelState,
    provider,
    model,
    perMessageQueueMode: dr.perMessageQueueMode,
    perMessageQueueOptions: dr.perMessageQueueOptions,
    typing: mw.typing,
    opts: runReplyOpts,
    defaultProvider: mw.defaultProvider,
    defaultModel: mw.defaultModel,
    timeoutMs: mw.timeoutMs,
    isNewSession,
    resetTriggered,
    systemSent,
    sessionEntry,
    sessionStore,
    sessionKey,
    sessionId,
    storePath,
    workspaceDir: mw.workspaceDir,
    abortedLastRun,
    autoRoutingThinking: mw.autoRoutingThinking,
  });
}

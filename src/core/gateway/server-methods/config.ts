import { resolveAgentWorkspaceDir, resolveDefaultAgentId } from "../../../agents/agent-scope.js";
import { listChannelPlugins } from "../../../channels/plugins/index.js";
import {
  formatDoctorNonInteractiveHint,
  type RestartSentinelPayload,
  writeRestartSentinel,
} from "../../../infra/restart-sentinel.js";
import { scheduleGatewaySigusr1Restart } from "../../../infra/restart.js";
import { appendLedgerEvent, type LedgerAppendEventParams } from "../../../ledger/event-store.js";
import { loadMarvPlugins } from "../../../plugins/loader.js";
import {
  CONFIG_PATH,
  loadConfig,
  parseConfigJson5,
  readConfigFileSnapshot,
  readConfigFileSnapshotForWrite,
  resolveConfigSnapshotHash,
  validateConfigObjectWithPlugins,
  writeConfigFile,
} from "../../config/config.js";
import { applyLegacyMigrations } from "../../config/legacy.js";
import { applyMergePatch } from "../../config/merge-patch.js";
import {
  redactConfigObject,
  redactConfigSnapshot,
  restoreRedactedValues,
} from "../../config/redact-snapshot.js";
import { buildConfigSchema, type ConfigSchemaResponse } from "../../config/schema.js";
import {
  buildSemanticConfigConversationId,
  createSemanticConfigRevision,
  createSemanticPatchProposal,
  findCommittedRevisionByProposalId,
  getSemanticConfigRevision,
  getSemanticPatchProposal,
  listSemanticConfigRevisions,
  updateSemanticConfigRevisionStatus,
  updateSemanticPatchProposalStatus,
  type SemanticConfigRevision,
  type SemanticPatchProposal,
} from "../../config/semantic-patches.js";
import { extractDeliveryInfo } from "../../config/sessions.js";
import type { MarvConfig } from "../../config/types.marv.js";
import {
  ErrorCodes,
  errorShape,
  validateConfigApplyParams,
  validateConfigGetParams,
  validateConfigPatchesCommitParams,
  validateConfigPatchesProposeParams,
  validateConfigPatchParams,
  validateConfigRevisionsListParams,
  validateConfigRevisionsRollbackParams,
  validateConfigSchemaParams,
  validateConfigSetParams,
} from "../protocol/index.js";
import { resolveBaseHashParam } from "./base-hash.js";
import { parseRestartRequestParams } from "./restart-request.js";
import type { GatewayRequestHandlers, RespondFn } from "./types.js";
import { assertValidParams } from "./validation.js";

function requireConfigBaseHash(
  params: unknown,
  snapshot: Awaited<ReturnType<typeof readConfigFileSnapshot>>,
  respond: RespondFn,
): boolean {
  if (!snapshot.exists) {
    return true;
  }
  const snapshotHash = resolveConfigSnapshotHash(snapshot);
  if (!snapshotHash) {
    respond(
      false,
      undefined,
      errorShape(
        ErrorCodes.INVALID_REQUEST,
        "config base hash unavailable; re-run config.get and retry",
      ),
    );
    return false;
  }
  const baseHash = resolveBaseHashParam(params);
  if (!baseHash) {
    respond(
      false,
      undefined,
      errorShape(
        ErrorCodes.INVALID_REQUEST,
        "config base hash required; re-run config.get and retry",
      ),
    );
    return false;
  }
  if (baseHash !== snapshotHash) {
    respond(
      false,
      undefined,
      errorShape(
        ErrorCodes.INVALID_REQUEST,
        "config changed since last load; re-run config.get and retry",
      ),
    );
    return false;
  }
  return true;
}

function parseRawConfigOrRespond(
  params: unknown,
  requestName: string,
  respond: RespondFn,
): string | null {
  const rawValue = (params as { raw?: unknown }).raw;
  if (typeof rawValue !== "string") {
    respond(
      false,
      undefined,
      errorShape(
        ErrorCodes.INVALID_REQUEST,
        `invalid ${requestName} params: raw (string) required`,
      ),
    );
    return null;
  }
  return rawValue;
}

function parseValidateConfigFromRawOrRespond(
  params: unknown,
  requestName: string,
  snapshot: Awaited<ReturnType<typeof readConfigFileSnapshot>>,
  respond: RespondFn,
): { config: MarvConfig; schema: ConfigSchemaResponse } | null {
  const rawValue = parseRawConfigOrRespond(params, requestName, respond);
  if (!rawValue) {
    return null;
  }
  const parsedRes = parseConfigJson5(rawValue);
  if (!parsedRes.ok) {
    respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, parsedRes.error));
    return null;
  }
  const schema = loadSchemaWithPlugins();
  const restored = restoreRedactedValues(parsedRes.parsed, snapshot.config, schema.uiHints);
  if (!restored.ok) {
    respond(
      false,
      undefined,
      errorShape(ErrorCodes.INVALID_REQUEST, restored.humanReadableMessage ?? "invalid config"),
    );
    return null;
  }
  const validated = validateConfigObjectWithPlugins(restored.result);
  if (!validated.ok) {
    respond(
      false,
      undefined,
      errorShape(ErrorCodes.INVALID_REQUEST, "invalid config", {
        details: { issues: validated.issues },
      }),
    );
    return null;
  }
  return { config: validated.config, schema };
}

function parsePatchObjectOrRespond(
  params: unknown,
  requestName: string,
  respond: RespondFn,
): Record<string, unknown> | null {
  const rawValue = (params as { raw?: unknown }).raw;
  if (typeof rawValue !== "string") {
    respond(
      false,
      undefined,
      errorShape(
        ErrorCodes.INVALID_REQUEST,
        `invalid ${requestName} params: raw (string) required`,
      ),
    );
    return null;
  }
  const parsedRes = parseConfigJson5(rawValue);
  if (!parsedRes.ok) {
    respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, parsedRes.error));
    return null;
  }
  if (
    !parsedRes.parsed ||
    typeof parsedRes.parsed !== "object" ||
    Array.isArray(parsedRes.parsed)
  ) {
    respond(
      false,
      undefined,
      errorShape(ErrorCodes.INVALID_REQUEST, `${requestName} raw must be an object`),
    );
    return null;
  }
  return parsedRes.parsed as Record<string, unknown>;
}

async function mergePatchAndWriteConfigOrRespond(params: {
  snapshot: Awaited<ReturnType<typeof readConfigFileSnapshot>>;
  writeOptions: Awaited<ReturnType<typeof readConfigFileSnapshotForWrite>>["writeOptions"];
  patch: Record<string, unknown>;
  respond: RespondFn;
}): Promise<{ config: MarvConfig; schema: ConfigSchemaResponse } | null> {
  if (!params.snapshot.valid) {
    params.respond(
      false,
      undefined,
      errorShape(ErrorCodes.INVALID_REQUEST, "invalid config; fix before patching"),
    );
    return null;
  }
  const merged = applyMergePatch(params.snapshot.config, params.patch, {
    mergeObjectArraysById: true,
  });
  const schema = loadSchemaWithPlugins();
  const restoredMerge = restoreRedactedValues(merged, params.snapshot.config, schema.uiHints);
  if (!restoredMerge.ok) {
    params.respond(
      false,
      undefined,
      errorShape(
        ErrorCodes.INVALID_REQUEST,
        restoredMerge.humanReadableMessage ?? "invalid config",
      ),
    );
    return null;
  }
  const migrated = applyLegacyMigrations(restoredMerge.result);
  const resolved = migrated.next ?? restoredMerge.result;
  const validated = validateConfigObjectWithPlugins(resolved);
  if (!validated.ok) {
    params.respond(
      false,
      undefined,
      errorShape(ErrorCodes.INVALID_REQUEST, "invalid config", {
        details: { issues: validated.issues },
      }),
    );
    return null;
  }
  await writeConfigFile(validated.config, params.writeOptions);
  return { config: validated.config, schema };
}

function resolveConfigRestartRequest(params: unknown): {
  sessionKey: string | undefined;
  note: string | undefined;
  restartDelayMs: number | undefined;
  deliveryContext: ReturnType<typeof extractDeliveryInfo>["deliveryContext"];
  threadId: ReturnType<typeof extractDeliveryInfo>["threadId"];
} {
  const { sessionKey, note, restartDelayMs } = parseRestartRequestParams(params);

  // Extract deliveryContext + threadId for routing after restart
  // Supports both :thread: (most channels) and :topic: (Telegram)
  const { deliveryContext, threadId } = extractDeliveryInfo(sessionKey);

  return {
    sessionKey,
    note,
    restartDelayMs,
    deliveryContext,
    threadId,
  };
}

function buildConfigRestartSentinelPayload(params: {
  kind: RestartSentinelPayload["kind"];
  mode: string;
  sessionKey: string | undefined;
  deliveryContext: ReturnType<typeof extractDeliveryInfo>["deliveryContext"];
  threadId: ReturnType<typeof extractDeliveryInfo>["threadId"];
  note: string | undefined;
}): RestartSentinelPayload {
  return {
    kind: params.kind,
    status: "ok",
    ts: Date.now(),
    sessionKey: params.sessionKey,
    deliveryContext: params.deliveryContext,
    threadId: params.threadId,
    message: params.note ?? null,
    doctorHint: formatDoctorNonInteractiveHint(),
    stats: {
      mode: params.mode,
      root: CONFIG_PATH,
    },
  };
}

async function tryWriteRestartSentinelPayload(
  payload: RestartSentinelPayload,
): Promise<string | null> {
  try {
    return await writeRestartSentinel(payload);
  } catch {
    return null;
  }
}

function loadSchemaWithPlugins(): ConfigSchemaResponse {
  const cfg = loadConfig();
  const workspaceDir = resolveAgentWorkspaceDir(cfg, resolveDefaultAgentId(cfg));
  const pluginRegistry = loadMarvPlugins({
    config: cfg,
    cache: true,
    workspaceDir,
    logger: {
      info: () => {},
      warn: () => {},
      error: () => {},
      debug: () => {},
    },
  });
  // Note: We can't easily cache this, as there are no callback that can invalidate
  // our cache. However, both loadConfig() and loadMarvPlugins() already cache
  // their results, and buildConfigSchema() is just a cheap transformation.
  return buildConfigSchema({
    plugins: pluginRegistry.plugins.map((plugin) => ({
      id: plugin.id,
      name: plugin.name,
      description: plugin.description,
      configUiHints: plugin.configUiHints,
      configSchema: plugin.configJsonSchema,
    })),
    channels: listChannelPlugins().map((entry) => ({
      id: entry.id,
      label: entry.meta.label,
      description: entry.meta.blurb,
      configSchema: entry.configSchema?.schema,
      configUiHints: entry.configSchema?.uiHints,
    })),
  });
}

function readOptionalString(params: Record<string, unknown>, key: string): string | undefined {
  const value = params[key];
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed || undefined;
}

function readOptionalBoolean(params: Record<string, unknown>, key: string): boolean | undefined {
  const value = params[key];
  if (typeof value !== "boolean") {
    return undefined;
  }
  return value;
}

function normalizeScope(value: string | undefined, fallback: string): string {
  const normalized = value?.trim().toLowerCase();
  return normalized || fallback;
}

function resolveActorId(params: Record<string, unknown>, clientActorId?: string): string {
  return readOptionalString(params, "actorId") ?? clientActorId?.trim() ?? "gateway";
}

function resolveSessionConversationId(sessionKey: string | undefined, fallback: string): string {
  const normalized = sessionKey?.trim();
  return normalized || fallback;
}

function summarizePatchKeys(patch: Record<string, unknown>): string[] {
  return Object.keys(patch).toSorted();
}

function appendLedgerEventSafe(params: LedgerAppendEventParams): void {
  try {
    appendLedgerEvent(params);
  } catch {
    // Ledger writes are best-effort and must not break config writes.
  }
}

async function commitSemanticProposalOrRespond(params: {
  proposal: SemanticPatchProposal;
  actorId: string;
  requestParams: Record<string, unknown>;
  clientActorId: string | undefined;
  respond: RespondFn;
}): Promise<{
  revision: SemanticConfigRevision;
  config: MarvConfig;
  schema: ConfigSchemaResponse;
  restart: ReturnType<typeof scheduleGatewaySigusr1Restart>;
  sentinelPath: string | null;
  payload: RestartSentinelPayload;
} | null> {
  const { snapshot, writeOptions } = await readConfigFileSnapshotForWrite();
  const mergedResult = await mergePatchAndWriteConfigOrRespond({
    snapshot,
    writeOptions,
    patch: params.proposal.patch,
    respond: params.respond,
  });
  if (!mergedResult) {
    return null;
  }
  updateSemanticPatchProposalStatus({
    proposalId: params.proposal.proposalId,
    status: "committed",
  });
  const revision = createSemanticConfigRevision({
    proposalId: params.proposal.proposalId,
    scopeType: params.proposal.scopeType,
    scopeId: params.proposal.scopeId,
    actorId: params.actorId,
    patch: params.proposal.patch,
    explanation: params.proposal.explanation,
    riskLevel: params.proposal.riskLevel,
    status: "committed",
    beforeConfig: snapshot.config,
    afterConfig: mergedResult.config,
  });

  const { sessionKey, note, restartDelayMs, deliveryContext, threadId } =
    resolveConfigRestartRequest(params.requestParams);
  const payload = buildConfigRestartSentinelPayload({
    kind: "config-patch",
    mode: "config.patches.commit",
    sessionKey,
    deliveryContext,
    threadId,
    note,
  });
  const sentinelPath = await tryWriteRestartSentinelPayload(payload);
  const restart = scheduleGatewaySigusr1Restart({
    delayMs: restartDelayMs,
    reason: "config.patches.commit",
  });

  appendLedgerEventSafe({
    conversationId: buildSemanticConfigConversationId(
      params.proposal.scopeType,
      params.proposal.scopeId,
    ),
    type: "PatchCommittedEvent",
    actorId: params.clientActorId ?? params.actorId,
    payload: {
      proposalId: params.proposal.proposalId,
      revision: revision.revision,
      riskLevel: params.proposal.riskLevel,
      patchKeys: summarizePatchKeys(params.proposal.patch),
    },
  });
  return {
    revision,
    config: mergedResult.config,
    schema: mergedResult.schema,
    restart,
    sentinelPath,
    payload,
  };
}

export const configHandlers: GatewayRequestHandlers = {
  "config.get": async ({ params, respond }) => {
    if (!assertValidParams(params, validateConfigGetParams, "config.get", respond)) {
      return;
    }
    const snapshot = await readConfigFileSnapshot();
    const schema = loadSchemaWithPlugins();
    respond(true, redactConfigSnapshot(snapshot, schema.uiHints), undefined);
  },
  "config.schema": ({ params, respond }) => {
    if (!assertValidParams(params, validateConfigSchemaParams, "config.schema", respond)) {
      return;
    }
    respond(true, loadSchemaWithPlugins(), undefined);
  },
  "config.set": async ({ params, respond }) => {
    if (!assertValidParams(params, validateConfigSetParams, "config.set", respond)) {
      return;
    }
    const { snapshot, writeOptions } = await readConfigFileSnapshotForWrite();
    if (!requireConfigBaseHash(params, snapshot, respond)) {
      return;
    }
    const parsed = parseValidateConfigFromRawOrRespond(params, "config.set", snapshot, respond);
    if (!parsed) {
      return;
    }
    await writeConfigFile(parsed.config, writeOptions);
    respond(
      true,
      {
        ok: true,
        path: CONFIG_PATH,
        config: redactConfigObject(parsed.config, parsed.schema.uiHints),
      },
      undefined,
    );
  },
  "config.patch": async ({ params, respond, client }) => {
    if (!assertValidParams(params, validateConfigPatchParams, "config.patch", respond)) {
      return;
    }
    const { snapshot, writeOptions } = await readConfigFileSnapshotForWrite();
    if (!requireConfigBaseHash(params, snapshot, respond)) {
      return;
    }
    const patch = parsePatchObjectOrRespond(params, "config.patch", respond);
    if (!patch) {
      return;
    }
    const mergedResult = await mergePatchAndWriteConfigOrRespond({
      snapshot,
      writeOptions,
      patch,
      respond,
    });
    if (!mergedResult) {
      return;
    }

    const { sessionKey, note, restartDelayMs, deliveryContext, threadId } =
      resolveConfigRestartRequest(params);
    appendLedgerEventSafe({
      conversationId: resolveSessionConversationId(sessionKey, "config:global:gateway"),
      type: "ConfigPatchedEvent",
      actorId: client?.connect.client.id,
      payload: {
        mode: "config.patch",
        patchKeys: summarizePatchKeys(patch),
      },
    });
    const payload = buildConfigRestartSentinelPayload({
      kind: "config-patch",
      mode: "config.patch",
      sessionKey,
      deliveryContext,
      threadId,
      note,
    });
    const sentinelPath = await tryWriteRestartSentinelPayload(payload);
    const restart = scheduleGatewaySigusr1Restart({
      delayMs: restartDelayMs,
      reason: "config.patch",
    });
    respond(
      true,
      {
        ok: true,
        path: CONFIG_PATH,
        config: redactConfigObject(mergedResult.config, mergedResult.schema.uiHints),
        restart,
        sentinel: {
          path: sentinelPath,
          payload,
        },
      },
      undefined,
    );
  },
  "config.apply": async ({ params, respond, client }) => {
    if (!assertValidParams(params, validateConfigApplyParams, "config.apply", respond)) {
      return;
    }
    const { snapshot, writeOptions } = await readConfigFileSnapshotForWrite();
    if (!requireConfigBaseHash(params, snapshot, respond)) {
      return;
    }
    const parsed = parseValidateConfigFromRawOrRespond(params, "config.apply", snapshot, respond);
    if (!parsed) {
      return;
    }
    await writeConfigFile(parsed.config, writeOptions);

    const { sessionKey, note, restartDelayMs, deliveryContext, threadId } =
      resolveConfigRestartRequest(params);
    appendLedgerEventSafe({
      conversationId: resolveSessionConversationId(sessionKey, "config:global:gateway"),
      type: "ConfigAppliedEvent",
      actorId: client?.connect.client.id,
      payload: {
        mode: "config.apply",
        topLevelKeys: summarizePatchKeys(parsed.config as Record<string, unknown>),
      },
    });
    const payload = buildConfigRestartSentinelPayload({
      kind: "config-apply",
      mode: "config.apply",
      sessionKey,
      deliveryContext,
      threadId,
      note,
    });
    const sentinelPath = await tryWriteRestartSentinelPayload(payload);
    const restart = scheduleGatewaySigusr1Restart({
      delayMs: restartDelayMs,
      reason: "config.apply",
    });
    respond(
      true,
      {
        ok: true,
        path: CONFIG_PATH,
        config: redactConfigObject(parsed.config, parsed.schema.uiHints),
        restart,
        sentinel: {
          path: sentinelPath,
          payload,
        },
      },
      undefined,
    );
  },
  "config.patches.propose": async ({ params, respond, client }) => {
    if (
      !assertValidParams(
        params,
        validateConfigPatchesProposeParams,
        "config.patches.propose",
        respond,
      )
    ) {
      return;
    }
    const requestParams = params as Record<string, unknown>;
    const naturalLanguage = readOptionalString(requestParams, "naturalLanguage");
    if (!naturalLanguage) {
      respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "naturalLanguage required"));
      return;
    }
    const scopeType = normalizeScope(readOptionalString(requestParams, "scopeType"), "global");
    const scopeId = normalizeScope(readOptionalString(requestParams, "scopeId"), "gateway");
    const actorId = resolveActorId(requestParams, client?.connect.client.id);

    let proposal: SemanticPatchProposal;
    try {
      proposal = createSemanticPatchProposal({
        scopeType,
        scopeId,
        naturalLanguage,
        actorId,
      });
    } catch (err) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          err instanceof Error ? err.message : "failed to create patch proposal",
        ),
      );
      return;
    }

    appendLedgerEventSafe({
      conversationId: buildSemanticConfigConversationId(scopeType, scopeId),
      type: "PatchProposedEvent",
      actorId: client?.connect.client.id ?? actorId,
      payload: {
        proposalId: proposal.proposalId,
        riskLevel: proposal.riskLevel,
        needsApproval: proposal.needsApproval,
        patchKeys: summarizePatchKeys(proposal.patch),
      },
    });

    const autoCommit = readOptionalBoolean(requestParams, "autoCommit") ?? false;
    if (!autoCommit) {
      respond(
        true,
        {
          proposalId: proposal.proposalId,
          scopeType: proposal.scopeType,
          scopeId: proposal.scopeId,
          riskLevel: proposal.riskLevel,
          needsApproval: proposal.needsApproval,
          patch: proposal.patch,
          explanation: proposal.explanation,
          status: proposal.status,
        },
        undefined,
      );
      return;
    }

    const committed = await commitSemanticProposalOrRespond({
      proposal,
      actorId,
      requestParams,
      clientActorId: client?.connect.client.id,
      respond,
    });
    if (!committed) {
      return;
    }
    respond(
      true,
      {
        proposalId: proposal.proposalId,
        status: "committed",
        autoCommitted: true,
        revision: committed.revision.revision,
        riskLevel: proposal.riskLevel,
        effectiveConfig: redactConfigObject(committed.config, committed.schema.uiHints),
        restart: committed.restart,
        sentinel: {
          path: committed.sentinelPath,
          payload: committed.payload,
        },
      },
      undefined,
    );
  },
  "config.patches.commit": async ({ params, respond, client }) => {
    if (
      !assertValidParams(
        params,
        validateConfigPatchesCommitParams,
        "config.patches.commit",
        respond,
      )
    ) {
      return;
    }
    const requestParams = params as Record<string, unknown>;
    const proposalId = readOptionalString(requestParams, "proposalId");
    if (!proposalId) {
      respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "proposalId required"));
      return;
    }
    const proposal = getSemanticPatchProposal(proposalId);
    if (!proposal) {
      respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "proposal not found"));
      return;
    }
    if (proposal.status === "committed") {
      const revision = findCommittedRevisionByProposalId(proposal.proposalId);
      const snapshot = await readConfigFileSnapshot();
      const schema = loadSchemaWithPlugins();
      respond(
        true,
        {
          proposalId: proposal.proposalId,
          status: "committed",
          revision: revision?.revision ?? null,
          riskLevel: proposal.riskLevel,
          effectiveConfig: redactConfigObject(snapshot.config, schema.uiHints),
        },
        undefined,
      );
      return;
    }
    if (proposal.status !== "open") {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.INVALID_REQUEST, `proposal is not open: ${proposal.status}`),
      );
      return;
    }
    const actorId = resolveActorId(requestParams, client?.connect.client.id);
    const committed = await commitSemanticProposalOrRespond({
      proposal,
      actorId,
      requestParams,
      clientActorId: client?.connect.client.id,
      respond,
    });
    if (!committed) {
      return;
    }
    respond(
      true,
      {
        proposalId: proposal.proposalId,
        status: committed.revision.status,
        revision: committed.revision.revision,
        riskLevel: committed.revision.riskLevel,
        effectiveConfig: redactConfigObject(committed.config, committed.schema.uiHints),
        restart: committed.restart,
        sentinel: {
          path: committed.sentinelPath,
          payload: committed.payload,
        },
      },
      undefined,
    );
  },
  "config.revisions.rollback": async ({ params, respond, client }) => {
    if (
      !assertValidParams(
        params,
        validateConfigRevisionsRollbackParams,
        "config.revisions.rollback",
        respond,
      )
    ) {
      return;
    }
    const requestParams = params as Record<string, unknown>;
    const revisionId = readOptionalString(requestParams, "revision");
    if (!revisionId) {
      respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "revision required"));
      return;
    }
    const target = getSemanticConfigRevision(revisionId);
    if (!target) {
      respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "revision not found"));
      return;
    }
    if (target.status !== "committed") {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.INVALID_REQUEST, "only committed revisions can be rolled back"),
      );
      return;
    }
    if (!target.beforeConfig) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          "revision does not include beforeConfig; rollback unavailable",
        ),
      );
      return;
    }

    const validatedBefore = validateConfigObjectWithPlugins(target.beforeConfig);
    if (!validatedBefore.ok) {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.INVALID_REQUEST, "stored rollback snapshot is invalid", {
          details: { issues: validatedBefore.issues },
        }),
      );
      return;
    }

    const actorId = resolveActorId(requestParams, client?.connect.client.id);
    const { snapshot, writeOptions } = await readConfigFileSnapshotForWrite();
    await writeConfigFile(validatedBefore.config, writeOptions);

    updateSemanticConfigRevisionStatus({
      revision: target.revision,
      status: "rolled_back",
    });
    const rollbackRevision = createSemanticConfigRevision({
      proposalId: target.proposalId ?? undefined,
      scopeType: target.scopeType,
      scopeId: target.scopeId,
      actorId,
      patch: {
        rolledBackRevision: target.revision,
      },
      explanation: `Rollback ${target.revision}`,
      riskLevel: "L3",
      status: "rolled_back",
      beforeConfig: snapshot.config,
      afterConfig: validatedBefore.config,
    });

    const { sessionKey, note, restartDelayMs, deliveryContext, threadId } =
      resolveConfigRestartRequest(requestParams);
    const payload = buildConfigRestartSentinelPayload({
      kind: "config-patch",
      mode: "config.revisions.rollback",
      sessionKey,
      deliveryContext,
      threadId,
      note,
    });
    const sentinelPath = await tryWriteRestartSentinelPayload(payload);
    const restart = scheduleGatewaySigusr1Restart({
      delayMs: restartDelayMs,
      reason: "config.revisions.rollback",
    });

    appendLedgerEventSafe({
      conversationId: buildSemanticConfigConversationId(target.scopeType, target.scopeId),
      type: "PatchRolledBackEvent",
      actorId: client?.connect.client.id ?? actorId,
      payload: {
        revision: target.revision,
        rollbackRevision: rollbackRevision.revision,
      },
    });
    const schema = loadSchemaWithPlugins();
    respond(
      true,
      {
        rolledBack: target.revision,
        rollbackRevision: rollbackRevision.revision,
        effectiveConfig: redactConfigObject(validatedBefore.config, schema.uiHints),
        restart,
        sentinel: {
          path: sentinelPath,
          payload,
        },
      },
      undefined,
    );
  },
  "config.revisions.list": ({ params, respond }) => {
    if (
      !assertValidParams(
        params,
        validateConfigRevisionsListParams,
        "config.revisions.list",
        respond,
      )
    ) {
      return;
    }
    const requestParams = params as Record<string, unknown>;
    const revisions = listSemanticConfigRevisions({
      scopeType: readOptionalString(requestParams, "scopeType"),
      scopeId: readOptionalString(requestParams, "scopeId"),
      limit: typeof requestParams.limit === "number" ? Math.floor(requestParams.limit) : undefined,
    });
    respond(
      true,
      {
        count: revisions.length,
        revisions: revisions.map((revision) => ({
          revision: revision.revision,
          proposalId: revision.proposalId,
          scopeType: revision.scopeType,
          scopeId: revision.scopeId,
          createdAt: revision.createdAt,
          actorId: revision.actorId,
          patch: revision.patch,
          explanation: revision.explanation,
          riskLevel: revision.riskLevel,
          status: revision.status,
        })),
      },
      undefined,
    );
  },
};

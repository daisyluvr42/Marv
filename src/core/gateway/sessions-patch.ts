import { randomUUID } from "node:crypto";
import { resolveDefaultAgentId } from "../../agents/agent-scope.js";
import {
  findModelInCatalog,
  findReasoningModelForProvider,
  type ModelCatalogEntry,
} from "../../agents/model/model-catalog.js";
import {
  resolveAllowedModelRef,
  resolveDefaultModelForAgent,
  resolveSubagentConfiguredModelSelection,
} from "../../agents/model/model-selection.js";
import { normalizeGroupActivation } from "../../auto-reply/inbound/group-activation.js";
import { normalizeQueueDropPolicy, normalizeQueueMode } from "../../auto-reply/queue/normalize.js";
import {
  formatThinkingLevels,
  formatXHighModelHint,
  normalizeElevatedLevel,
  normalizeReasoningLevel,
  normalizeThinkLevel,
  normalizeUsageDisplay,
  supportsXHighThinking,
} from "../../auto-reply/support/thinking.js";
import {
  isSubagentSessionKey,
  normalizeAgentId,
  parseAgentSessionKey,
} from "../../routing/session-key.js";
import {
  normalizeSubagentAnnounceMode,
  normalizeSubagentMetadataValue,
} from "../../shared/subagent-metadata.js";
import type { MarvConfig } from "../config/config.js";
import type { SessionEntry } from "../config/sessions.js";
import { applyVerboseOverride, parseVerboseOverride } from "../session/level-overrides.js";
import { applyModelOverrideToSessionEntry } from "../session/model-overrides.js";
import { normalizeSendPolicy } from "../session/send-policy.js";
import { parseSessionLabel } from "../session/session-label.js";
import {
  ErrorCodes,
  type ErrorShape,
  errorShape,
  type SessionsPatchParams,
} from "./protocol/index.js";

function invalid(message: string): { ok: false; error: ErrorShape } {
  return { ok: false, error: errorShape(ErrorCodes.INVALID_REQUEST, message) };
}

function normalizeExecHost(raw: string): "sandbox" | "gateway" | "node" | undefined {
  const normalized = raw.trim().toLowerCase();
  if (normalized === "sandbox" || normalized === "gateway" || normalized === "node") {
    return normalized;
  }
  return undefined;
}

function normalizeExecSecurity(raw: string): "deny" | "allowlist" | "full" | undefined {
  const normalized = raw.trim().toLowerCase();
  if (normalized === "deny" || normalized === "allowlist" || normalized === "full") {
    return normalized;
  }
  return undefined;
}

function normalizeExecAsk(raw: string): "off" | "on-miss" | "always" | undefined {
  const normalized = raw.trim().toLowerCase();
  if (normalized === "off" || normalized === "on-miss" || normalized === "always") {
    return normalized;
  }
  return undefined;
}

function updateSubagentMetadataField(params: {
  patchValue: unknown;
  existingValue: string | undefined;
  storeKey: string;
  label: string;
  assign: (value: string) => void;
}): { ok: true } | { ok: false; error: ErrorShape } {
  if (params.patchValue === null) {
    if (params.existingValue) {
      return invalid(`${params.label} cannot be cleared once set`);
    }
    return { ok: true };
  }
  if (params.patchValue === undefined) {
    return { ok: true };
  }
  if (!isSubagentSessionKey(params.storeKey)) {
    return invalid(`${params.label} is only supported for subagent:* sessions`);
  }
  const normalized = normalizeSubagentMetadataValue(params.patchValue);
  if (!normalized) {
    return invalid(`invalid ${params.label}: empty`);
  }
  if (params.existingValue && params.existingValue !== normalized) {
    return invalid(`${params.label} cannot be changed once set`);
  }
  params.assign(normalized);
  return { ok: true };
}

export async function applySessionsPatchToStore(params: {
  cfg: MarvConfig;
  store: Record<string, SessionEntry>;
  storeKey: string;
  patch: SessionsPatchParams;
  loadGatewayModelCatalog?: () => Promise<ModelCatalogEntry[]>;
}): Promise<
  { ok: true; entry: SessionEntry; notices?: string[] } | { ok: false; error: ErrorShape }
> {
  const { cfg, store, storeKey, patch } = params;
  const now = Date.now();
  const parsedAgent = parseAgentSessionKey(storeKey);
  const sessionAgentId = normalizeAgentId(parsedAgent?.agentId ?? resolveDefaultAgentId(cfg));
  const resolvedDefault = resolveDefaultModelForAgent({ cfg, agentId: sessionAgentId });
  const subagentModelHint = isSubagentSessionKey(storeKey)
    ? resolveSubagentConfiguredModelSelection({ cfg, agentId: sessionAgentId })
    : undefined;
  const existing = store[storeKey];
  const next: SessionEntry = existing
    ? {
        ...existing,
        updatedAt: Math.max(existing.updatedAt ?? 0, now),
      }
    : { sessionId: randomUUID(), updatedAt: now };

  if ("spawnedBy" in patch) {
    const raw = patch.spawnedBy;
    if (raw === null) {
      if (existing?.spawnedBy) {
        return invalid("spawnedBy cannot be cleared once set");
      }
    } else if (raw !== undefined) {
      const trimmed = String(raw).trim();
      if (!trimmed) {
        return invalid("invalid spawnedBy: empty");
      }
      if (!isSubagentSessionKey(storeKey)) {
        return invalid("spawnedBy is only supported for subagent:* sessions");
      }
      if (existing?.spawnedBy && existing.spawnedBy !== trimmed) {
        return invalid("spawnedBy cannot be changed once set");
      }
      next.spawnedBy = trimmed;
    }
  }

  if ("spawnDepth" in patch) {
    const raw = patch.spawnDepth;
    if (raw === null) {
      if (typeof existing?.spawnDepth === "number") {
        return invalid("spawnDepth cannot be cleared once set");
      }
    } else if (raw !== undefined) {
      if (!isSubagentSessionKey(storeKey)) {
        return invalid("spawnDepth is only supported for subagent:* sessions");
      }
      const numeric = Number(raw);
      if (!Number.isInteger(numeric) || numeric < 0) {
        return invalid("invalid spawnDepth (use an integer >= 0)");
      }
      const normalized = numeric;
      if (typeof existing?.spawnDepth === "number" && existing.spawnDepth !== normalized) {
        return invalid("spawnDepth cannot be changed once set");
      }
      next.spawnDepth = normalized;
    }
  }

  if ("subagentRole" in patch) {
    const result = updateSubagentMetadataField({
      patchValue: patch.subagentRole,
      existingValue: existing?.subagentRole,
      storeKey,
      label: "subagentRole",
      assign: (value) => {
        next.subagentRole = value;
      },
    });
    if (!result.ok) {
      return result;
    }
  }

  if ("subagentPreset" in patch) {
    const result = updateSubagentMetadataField({
      patchValue: patch.subagentPreset,
      existingValue: existing?.subagentPreset,
      storeKey,
      label: "subagentPreset",
      assign: (value) => {
        next.subagentPreset = value;
      },
    });
    if (!result.ok) {
      return result;
    }
  }

  if ("subagentTaskGroup" in patch) {
    const result = updateSubagentMetadataField({
      patchValue: patch.subagentTaskGroup,
      existingValue: existing?.subagentTaskGroup,
      storeKey,
      label: "subagentTaskGroup",
      assign: (value) => {
        next.subagentTaskGroup = value;
      },
    });
    if (!result.ok) {
      return result;
    }
  }

  if ("subagentDispatchId" in patch) {
    const result = updateSubagentMetadataField({
      patchValue: patch.subagentDispatchId,
      existingValue: existing?.subagentDispatchId,
      storeKey,
      label: "subagentDispatchId",
      assign: (value) => {
        next.subagentDispatchId = value;
      },
    });
    if (!result.ok) {
      return result;
    }
  }

  if ("subagentAnnounceMode" in patch) {
    const raw = patch.subagentAnnounceMode;
    if (raw === null) {
      if (existing?.subagentAnnounceMode) {
        return invalid("subagentAnnounceMode cannot be cleared once set");
      }
    } else if (raw !== undefined) {
      if (!isSubagentSessionKey(storeKey)) {
        return invalid("subagentAnnounceMode is only supported for subagent:* sessions");
      }
      const normalized = normalizeSubagentAnnounceMode(raw);
      if (!normalized) {
        return invalid('invalid subagentAnnounceMode (use "child"|"aggregate")');
      }
      if (existing?.subagentAnnounceMode && existing.subagentAnnounceMode !== normalized) {
        return invalid("subagentAnnounceMode cannot be changed once set");
      }
      next.subagentAnnounceMode = normalized;
    }
  }

  if ("label" in patch) {
    const raw = patch.label;
    if (raw === null) {
      delete next.label;
    } else if (raw !== undefined) {
      const parsed = parseSessionLabel(raw);
      if (!parsed.ok) {
        return invalid(parsed.error);
      }
      for (const [key, entry] of Object.entries(store)) {
        if (key === storeKey) {
          continue;
        }
        if (entry?.label === parsed.label) {
          return invalid(`label already in use: ${parsed.label}`);
        }
      }
      next.label = parsed.label;
    }
  }

  if ("thinkingLevel" in patch) {
    const raw = patch.thinkingLevel;
    if (raw === null) {
      // Clear the override and fall back to model default
      delete next.thinkingLevel;
    } else if (raw !== undefined) {
      const normalized = normalizeThinkLevel(String(raw));
      if (!normalized) {
        const hintProvider = existing?.providerOverride?.trim() || resolvedDefault.provider;
        const hintModel = existing?.modelOverride?.trim() || resolvedDefault.model;
        return invalid(
          `invalid thinkingLevel (use ${formatThinkingLevels(hintProvider, hintModel, "|")})`,
        );
      }
      next.thinkingLevel = normalized;
    }
  }

  if ("verboseLevel" in patch) {
    const raw = patch.verboseLevel;
    const parsed = parseVerboseOverride(raw);
    if (!parsed.ok) {
      return invalid(parsed.error);
    }
    applyVerboseOverride(next, parsed.value);
  }

  if ("reasoningLevel" in patch) {
    const raw = patch.reasoningLevel;
    if (raw === null) {
      delete next.reasoningLevel;
    } else if (raw !== undefined) {
      const normalized = normalizeReasoningLevel(String(raw));
      if (!normalized) {
        return invalid('invalid reasoningLevel (use "on"|"off"|"stream")');
      }
      if (normalized === "off") {
        delete next.reasoningLevel;
      } else {
        next.reasoningLevel = normalized;
      }
    }
  }

  if ("responseUsage" in patch) {
    const raw = patch.responseUsage;
    if (raw === null) {
      delete next.responseUsage;
    } else if (raw !== undefined) {
      const normalized = normalizeUsageDisplay(String(raw));
      if (!normalized) {
        return invalid('invalid responseUsage (use "off"|"tokens"|"full")');
      }
      if (normalized === "off") {
        delete next.responseUsage;
      } else {
        next.responseUsage = normalized;
      }
    }
  }

  if ("elevatedLevel" in patch) {
    const raw = patch.elevatedLevel;
    if (raw === null) {
      delete next.elevatedLevel;
    } else if (raw !== undefined) {
      const normalized = normalizeElevatedLevel(String(raw));
      if (!normalized) {
        return invalid('invalid elevatedLevel (use "on"|"off"|"ask"|"full")');
      }
      // Persist "off" explicitly so patches can override defaults.
      next.elevatedLevel = normalized;
    }
  }

  if ("execHost" in patch) {
    const raw = patch.execHost;
    if (raw === null) {
      delete next.execHost;
    } else if (raw !== undefined) {
      const normalized = normalizeExecHost(String(raw));
      if (!normalized) {
        return invalid('invalid execHost (use "sandbox"|"gateway"|"node")');
      }
      next.execHost = normalized;
    }
  }

  if ("execSecurity" in patch) {
    const raw = patch.execSecurity;
    if (raw === null) {
      delete next.execSecurity;
    } else if (raw !== undefined) {
      const normalized = normalizeExecSecurity(String(raw));
      if (!normalized) {
        return invalid('invalid execSecurity (use "deny"|"allowlist"|"full")');
      }
      next.execSecurity = normalized;
    }
  }

  if ("execAsk" in patch) {
    const raw = patch.execAsk;
    if (raw === null) {
      delete next.execAsk;
    } else if (raw !== undefined) {
      const normalized = normalizeExecAsk(String(raw));
      if (!normalized) {
        return invalid('invalid execAsk (use "off"|"on-miss"|"always")');
      }
      next.execAsk = normalized;
    }
  }

  if ("execNode" in patch) {
    const raw = patch.execNode;
    if (raw === null) {
      delete next.execNode;
    } else if (raw !== undefined) {
      const trimmed = String(raw).trim();
      if (!trimmed) {
        return invalid("invalid execNode: empty");
      }
      next.execNode = trimmed;
    }
  }

  if ("queueMode" in patch) {
    const raw = patch.queueMode;
    if (raw === null) {
      delete next.queueMode;
    } else if (raw !== undefined) {
      const normalized = normalizeQueueMode(String(raw));
      if (!normalized) {
        return invalid(
          'invalid queueMode (use "steer"|"followup"|"collect"|"steer-backlog"|"interrupt")',
        );
      }
      next.queueMode = normalized;
    }
  }

  if ("queueDebounceMs" in patch) {
    const raw = patch.queueDebounceMs;
    if (raw === null) {
      delete next.queueDebounceMs;
    } else if (raw !== undefined) {
      const numeric = Number(raw);
      if (!Number.isFinite(numeric) || numeric < 0) {
        return invalid("invalid queueDebounceMs (use a number >= 0)");
      }
      next.queueDebounceMs = Math.floor(numeric);
    }
  }

  if ("queueCap" in patch) {
    const raw = patch.queueCap;
    if (raw === null) {
      delete next.queueCap;
    } else if (raw !== undefined) {
      const numeric = Number(raw);
      if (!Number.isFinite(numeric) || numeric < 1) {
        return invalid("invalid queueCap (use an integer >= 1)");
      }
      next.queueCap = Math.floor(numeric);
    }
  }

  if ("queueDrop" in patch) {
    const raw = patch.queueDrop;
    if (raw === null) {
      delete next.queueDrop;
    } else if (raw !== undefined) {
      const normalized = normalizeQueueDropPolicy(String(raw));
      if (!normalized) {
        return invalid('invalid queueDrop (use "old"|"new"|"summarize")');
      }
      next.queueDrop = normalized;
    }
  }

  if ("model" in patch) {
    const raw = patch.model;
    if (raw === null) {
      applyModelOverrideToSessionEntry({
        entry: next,
        selection: {
          provider: resolvedDefault.provider,
          model: resolvedDefault.model,
          isDefault: true,
        },
      });
    } else if (raw !== undefined) {
      const trimmed = String(raw).trim();
      if (!trimmed) {
        return invalid("invalid model: empty");
      }
      if (!params.loadGatewayModelCatalog) {
        return {
          ok: false,
          error: errorShape(ErrorCodes.UNAVAILABLE, "model catalog unavailable"),
        };
      }
      const catalog = await params.loadGatewayModelCatalog();
      const resolved = resolveAllowedModelRef({
        cfg,
        catalog,
        raw: trimmed,
        defaultProvider: resolvedDefault.provider,
        defaultModel: subagentModelHint ?? resolvedDefault.model,
      });
      if ("error" in resolved) {
        return invalid(resolved.error);
      }
      // When the user explicitly names a model — even if it matches the configured
      // default — treat it as a deliberate pin so the session stays on that model.
      // The `isDefault: true` (clear-override) path is already handled above when
      // `raw === null`, which is the explicit "reset to default" signal.
      applyModelOverrideToSessionEntry({
        entry: next,
        selection: {
          provider: resolved.ref.provider,
          model: resolved.ref.model,
          isDefault: false,
        },
      });
    }
  }

  // --- Sync thinking level ↔ model ---
  const notices: string[] = [];

  if ("model" in patch && !("thinkingLevel" in patch)) {
    // Model changed; adapt thinking level silently.
    const effectiveProvider = next.providerOverride ?? resolvedDefault.provider;
    const effectiveModel = next.modelOverride ?? resolvedDefault.model;
    const catalog = params.loadGatewayModelCatalog
      ? await params.loadGatewayModelCatalog()
      : undefined;
    if (catalog) {
      const catalogEntry = findModelInCatalog(catalog, effectiveProvider, effectiveModel);
      if (catalogEntry?.reasoning === true) {
        const currentThink = next.thinkingLevel ?? "off";
        if (currentThink === "off") {
          next.thinkingLevel = "low";
          notices.push(`Thinking set to low (default for ${effectiveModel})`);
        }
      } else if (next.thinkingLevel && next.thinkingLevel !== "off") {
        delete next.thinkingLevel;
        notices.push(`Thinking set to off (${effectiveModel} does not support reasoning)`);
      }
    }
  }

  if ("thinkingLevel" in patch && !("model" in patch)) {
    // Thinking level changed; auto-switch model if current model is non-reasoning.
    const normalized = next.thinkingLevel;
    if (normalized && normalized !== "off") {
      const effectiveProvider = next.providerOverride ?? resolvedDefault.provider;
      const effectiveModel = next.modelOverride ?? resolvedDefault.model;
      const catalog = params.loadGatewayModelCatalog
        ? await params.loadGatewayModelCatalog()
        : undefined;
      if (catalog) {
        const catalogEntry = findModelInCatalog(catalog, effectiveProvider, effectiveModel);
        if (!catalogEntry?.reasoning) {
          const reasoningModel = findReasoningModelForProvider(catalog, effectiveProvider);
          if (reasoningModel) {
            // Auto-switch to a reasoning model is a deliberate selection;
            // preserve it as a manual override so later runs don't drift.
            applyModelOverrideToSessionEntry({
              entry: next,
              selection: {
                provider: reasoningModel.provider,
                model: reasoningModel.id,
                isDefault: false,
              },
            });
            notices.push(
              `Model switched to ${reasoningModel.provider}/${reasoningModel.id} (reasoning-capable)`,
            );
          } else {
            notices.push(
              `Warning: no reasoning-capable ${effectiveProvider} model found. Switch models for thinking support.`,
            );
          }
        }
      }
    }
  }

  if (next.thinkingLevel === "xhigh") {
    const effectiveProvider = next.providerOverride ?? resolvedDefault.provider;
    const effectiveModel = next.modelOverride ?? resolvedDefault.model;
    if (!supportsXHighThinking(effectiveProvider, effectiveModel)) {
      if ("thinkingLevel" in patch) {
        return invalid(`thinkingLevel "xhigh" is only supported for ${formatXHighModelHint()}`);
      }
      next.thinkingLevel = "high";
    }
  }

  if ("sendPolicy" in patch) {
    const raw = patch.sendPolicy;
    if (raw === null) {
      delete next.sendPolicy;
    } else if (raw !== undefined) {
      const normalized = normalizeSendPolicy(String(raw));
      if (!normalized) {
        return invalid('invalid sendPolicy (use "allow"|"deny")');
      }
      next.sendPolicy = normalized;
    }
  }

  if ("groupActivation" in patch) {
    const raw = patch.groupActivation;
    if (raw === null) {
      delete next.groupActivation;
    } else if (raw !== undefined) {
      const normalized = normalizeGroupActivation(String(raw));
      if (!normalized) {
        return invalid('invalid groupActivation (use "mention"|"always")');
      }
      next.groupActivation = normalized;
    }
  }

  store[storeKey] = next;
  return { ok: true, entry: next, ...(notices.length > 0 ? { notices } : {}) };
}

import type { AgentModelListConfig } from "../../core/config/types.agent-defaults.js";
import type { MarvConfig } from "../../core/config/types.js";
import { resolveAgentConfig } from "../agent-scope.js";
import { DEFAULT_MODEL, DEFAULT_PROVIDER } from "../defaults.js";
import { modelKey, normalizeModelRef, parseModelRef, type ModelRef } from "./model-resolve.js";

export type OrderedModelRouteEntry = ModelRef & {
  ref: string;
  source: "current" | "configured";
};

export type OrderedModelRoutePlan = {
  entries: OrderedModelRouteEntry[];
  hasConfiguredRoute: boolean;
};

function normalizeModelListConfig(value: unknown): AgentModelListConfig | undefined {
  if (typeof value === "string") {
    const primary = value.trim();
    return primary ? { primary } : undefined;
  }
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const raw = value as { primary?: unknown; fallbacks?: unknown };
  const primary = typeof raw.primary === "string" ? raw.primary.trim() : undefined;
  const fallbacks = Array.isArray(raw.fallbacks)
    ? raw.fallbacks.map((entry) => (typeof entry === "string" ? entry.trim() : "")).filter(Boolean)
    : undefined;
  if (!primary && (!fallbacks || fallbacks.length === 0)) {
    return undefined;
  }
  return {
    ...(primary ? { primary } : {}),
    ...(fallbacks && fallbacks.length > 0 ? { fallbacks } : {}),
  };
}

export function resolveConfiguredModelList(params: {
  cfg?: MarvConfig;
  agentId?: string;
}): AgentModelListConfig | undefined {
  if (!params.cfg) {
    return undefined;
  }
  const agentConfig = params.agentId ? resolveAgentConfig(params.cfg, params.agentId) : undefined;
  return (
    normalizeModelListConfig(agentConfig?.model) ??
    normalizeModelListConfig(params.cfg.agents?.defaults?.model)
  );
}

export function resolveOrderedModelRoutePlan(params: {
  cfg?: MarvConfig;
  agentId?: string;
  primary?: ModelRef;
  defaultProvider?: string;
  includeDefaultWhenEmpty?: boolean;
}): OrderedModelRoutePlan {
  const defaultProvider = params.defaultProvider ?? DEFAULT_PROVIDER;
  const configured = resolveConfiguredModelList({
    cfg: params.cfg,
    agentId: params.agentId,
  });
  const entries: OrderedModelRouteEntry[] = [];
  const seen = new Set<string>();

  const addRef = (ref: ModelRef, source: OrderedModelRouteEntry["source"]) => {
    const normalized = normalizeModelRef(ref.provider, ref.model);
    const key = modelKey(normalized.provider, normalized.model);
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    entries.push({
      ...normalized,
      ref: key,
      source,
    });
  };

  const addRaw = (raw: string | undefined, source: OrderedModelRouteEntry["source"]) => {
    if (!raw?.trim()) {
      return;
    }
    const parsed = parseModelRef(raw, defaultProvider);
    if (parsed) {
      addRef(parsed, source);
    }
  };

  if (params.primary) {
    addRef(params.primary, "current");
  }

  addRaw(configured?.primary, "configured");
  for (const fallback of configured?.fallbacks ?? []) {
    addRaw(fallback, "configured");
  }

  if (entries.length === 0 && params.includeDefaultWhenEmpty) {
    addRef({ provider: DEFAULT_PROVIDER, model: DEFAULT_MODEL }, "configured");
  }

  return {
    entries,
    hasConfiguredRoute: Boolean(configured?.primary || configured?.fallbacks?.length),
  };
}

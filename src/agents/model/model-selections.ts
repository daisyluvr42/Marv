import type { MarvConfig } from "../../core/config/config.js";
import { loadConfig } from "../../core/config/config.js";
import { parseModelRef } from "./model-selection.js";

const LOCAL_PROVIDER_FAMILY = ["local", "ollama", "vllm", "lmstudio", "localai", "llamacpp"];

function normalizeProvider(value: string): string {
  return value.trim().toLowerCase();
}

export function resolveProviderFamilyProviders(provider: string): Set<string> {
  const normalized = normalizeProvider(provider);
  if (!normalized) {
    return new Set();
  }
  if (normalized === "google" || normalized === "google-gemini-cli") {
    return new Set(["google", "google-gemini-cli"]);
  }
  if (LOCAL_PROVIDER_FAMILY.includes(normalized)) {
    return new Set(LOCAL_PROVIDER_FAMILY);
  }
  return new Set([normalized]);
}

function resolveSelectionSourceProviders(cfg: MarvConfig, sourceKey: string): Set<string> {
  const trimmed = sourceKey.trim();
  if (!trimmed) {
    return new Set();
  }
  const profileProvider = cfg.auth?.profiles?.[trimmed]?.provider;
  if (profileProvider?.trim()) {
    return resolveProviderFamilyProviders(profileProvider);
  }
  return resolveProviderFamilyProviders(trimmed);
}

export function hasConfiguredModelSelections(cfg: MarvConfig): boolean {
  return Object.keys(cfg.models?.selections ?? {}).length > 0;
}

export function resolveSelectedModelRefs(params: {
  cfg: MarvConfig;
  defaultProvider: string;
}): Set<string> {
  const selections = params.cfg.models?.selections ?? {};
  const refs = new Set<string>();

  for (const [sourceKey, sourceRefs] of Object.entries(selections)) {
    const providerFamily = resolveSelectionSourceProviders(params.cfg, sourceKey);
    const profileProvider = params.cfg.auth?.profiles?.[sourceKey]?.provider;
    const preferredProvider = profileProvider ? normalizeProvider(profileProvider) : null;
    if (providerFamily.size === 0 || !Array.isArray(sourceRefs)) {
      continue;
    }
    for (const rawRef of sourceRefs) {
      const parsed = parseModelRef(String(rawRef ?? ""), params.defaultProvider);
      if (!parsed) {
        continue;
      }
      const modelProvider = normalizeProvider(parsed.provider);
      if (!providerFamily.has(modelProvider)) {
        continue;
      }
      const effectiveProvider =
        preferredProvider &&
        providerFamily.has(preferredProvider) &&
        modelProvider !== preferredProvider
          ? preferredProvider
          : parsed.provider;
      refs.add(`${effectiveProvider}/${parsed.model}`);
    }
  }

  return refs;
}

function normalizeModelRefsForSelection(
  refs: readonly string[],
  defaultProvider: string,
): string[] {
  const normalized: string[] = [];
  const seen = new Set<string>();
  for (const rawRef of refs) {
    const parsed = parseModelRef(String(rawRef ?? ""), defaultProvider);
    if (!parsed) {
      continue;
    }
    const ref = `${parsed.provider}/${parsed.model}`;
    if (seen.has(ref)) {
      continue;
    }
    seen.add(ref);
    normalized.push(ref);
  }
  return normalized;
}

function groupModelRefsByProvider(params: {
  refs: readonly string[];
  defaultProvider: string;
}): Record<string, string[]> {
  const grouped = new Map<string, string[]>();
  for (const ref of normalizeModelRefsForSelection(params.refs, params.defaultProvider)) {
    const parsed = parseModelRef(ref, params.defaultProvider);
    if (!parsed) {
      continue;
    }
    const provider = parsed.provider;
    const existing = grouped.get(provider);
    if (existing) {
      existing.push(ref);
      continue;
    }
    grouped.set(provider, [ref]);
  }

  return Object.fromEntries(grouped);
}

export function setSelectedModelRefsForSource(params: {
  cfg: MarvConfig;
  sourceKey: string;
  refs: readonly string[];
  defaultProvider: string;
}): MarvConfig {
  const sourceKey = params.sourceKey.trim();
  if (!sourceKey) {
    return params.cfg;
  }
  const nextRefs = normalizeModelRefsForSelection(params.refs, params.defaultProvider);
  const nextSelections = { ...params.cfg.models?.selections };
  if (nextRefs.length > 0) {
    nextSelections[sourceKey] = nextRefs;
  } else {
    delete nextSelections[sourceKey];
  }

  return {
    ...params.cfg,
    models: {
      ...params.cfg.models,
      selections: Object.keys(nextSelections).length > 0 ? nextSelections : undefined,
    },
  };
}

export function syncProviderSelectionsFromProviderConfig(params: {
  cfg: MarvConfig;
  providerId: string;
  defaultProvider?: string;
}): MarvConfig {
  const providerId = params.providerId.trim();
  if (!providerId) {
    return params.cfg;
  }
  const refs =
    params.cfg.models?.providers?.[providerId]?.models?.map(
      (model) => `${providerId}/${model.id}`,
    ) ?? [];
  return setSelectedModelRefsForSource({
    cfg: params.cfg,
    sourceKey: providerId,
    refs,
    defaultProvider: params.defaultProvider ?? providerId,
  });
}

export function replaceSelectedModelRefsByProvider(params: {
  cfg: MarvConfig;
  refs: readonly string[];
  defaultProvider: string;
}): MarvConfig {
  const grouped = groupModelRefsByProvider({
    refs: params.refs,
    defaultProvider: params.defaultProvider,
  });

  return {
    ...params.cfg,
    models: {
      ...params.cfg.models,
      selections: Object.keys(grouped).length > 0 ? grouped : undefined,
    },
  };
}

/**
 * Remove a confirmed-unsupported model from `models.selections` so it
 * no longer appears in the /models picker or future fallback lists.
 * Writes directly to config file (side-effect).
 */
export function pruneUnsupportedModelFromSelections(params: {
  cfg: MarvConfig;
  provider: string;
  model: string;
}): void {
  const ref = `${params.provider}/${params.model}`;
  const selections = params.cfg.models?.selections;
  if (!selections) {
    return;
  }
  let changed = false;
  const updated = { ...selections };
  for (const [sourceKey, refs] of Object.entries(updated)) {
    if (!Array.isArray(refs)) {
      continue;
    }
    const filtered = refs.filter((r) => r !== ref);
    if (filtered.length < refs.length) {
      changed = true;
      if (filtered.length > 0) {
        updated[sourceKey] = filtered;
      } else {
        delete updated[sourceKey];
      }
    }
  }
  if (!changed) {
    return;
  }
  // Fire-and-forget: persist the pruned selections to config file.
  void (async () => {
    try {
      const { writeConfigFile } = await import("../../core/config/io.js");
      const freshConfig = loadConfig();
      const freshSelections = { ...freshConfig.models?.selections };
      for (const [sourceKey, refs] of Object.entries(freshSelections)) {
        if (!Array.isArray(refs)) {
          continue;
        }
        const filtered = refs.filter((r) => r !== ref);
        if (filtered.length < refs.length) {
          if (filtered.length > 0) {
            freshSelections[sourceKey] = filtered;
          } else {
            delete freshSelections[sourceKey];
          }
        }
      }
      await writeConfigFile({
        ...freshConfig,
        models: {
          ...freshConfig.models,
          selections: Object.keys(freshSelections).length > 0 ? freshSelections : undefined,
        },
      });
    } catch {
      // Best-effort: config write may fail if locked by another process.
    }
  })();
}

import type { MarvConfig } from "../../core/config/config.js";
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
    if (providerFamily.size === 0 || !Array.isArray(sourceRefs)) {
      continue;
    }
    for (const rawRef of sourceRefs) {
      const parsed = parseModelRef(String(rawRef ?? ""), params.defaultProvider);
      if (!parsed) {
        continue;
      }
      if (!providerFamily.has(normalizeProvider(parsed.provider))) {
        continue;
      }
      refs.add(`${parsed.provider}/${parsed.model}`);
    }
  }

  return refs;
}

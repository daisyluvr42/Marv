import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { MarvConfig } from "../../core/config/config.js";
import { resolveStateDir } from "../../core/config/paths.js";
import { ensureAuthProfileStore, listProfilesForProvider } from "../auth-profiles.js";
import { resolveApiKeyForProvider } from "./model-auth.js";
import { getCustomProviderApiKey, resolveEnvApiKey } from "./model-auth.js";
import type { ModelCatalogEntry } from "./model-catalog.js";
import { loadModelCatalog } from "./model-catalog.js";
import { normalizeProviderId, parseModelRef } from "./model-selection.js";

const REGISTRY_VERSION = 1;
const REGISTRY_REFRESH_INTERVAL_MS = 7 * 24 * 60 * 60 * 1000;
const REGISTRY_CHECK_INTERVAL_MS = 12 * 60 * 60 * 1000;
const REGISTRY_FILENAME = "model-registry.json";
const REGISTRY_DIRNAME = "runtime";
const LOCAL_PROVIDERS = new Set(["local", "ollama", "vllm", "lmstudio", "localai", "llamacpp"]);
let refreshTimer: NodeJS.Timeout | null = null;

export type RuntimeRegistryModelStatus = "active" | "deprecated" | "removed";
export type RuntimeRegistryModelSource = "baseline_catalog" | "official_api";

export type RuntimeRegistryModelCapability = "text" | "vision" | "coding" | "tools";
export type RuntimeRegistryModelTier = "low" | "standard" | "high";
export type RuntimeRegistryModelLocation = "local" | "cloud";

export type RuntimeRegistryModel = {
  ref: string;
  provider: string;
  model: string;
  displayName: string;
  source: RuntimeRegistryModelSource;
  status: RuntimeRegistryModelStatus;
  capabilities: RuntimeRegistryModelCapability[];
  tier: RuntimeRegistryModelTier;
  location: RuntimeRegistryModelLocation;
  contextWindow?: number;
  docsUrl?: string;
};

export type RuntimeRegistryProviderStatus = {
  provider: string;
  officialDocsUrls: string[];
  fetchMode: "official_api" | "baseline_only";
  lastAttemptAt?: number;
  lastSuccessAt?: number;
  lastError?: string;
  configured: boolean;
};

export type RuntimeModelRegistry = {
  version: number;
  generatedAt: number;
  lastSuccessfulRefreshAt?: number;
  nextRefreshAfter?: number;
  etag: string;
  providers: Record<string, RuntimeRegistryProviderStatus>;
  models: RuntimeRegistryModel[];
};

type ProviderGuide = {
  provider: string;
  officialDocsUrls: string[];
  fetchMode: "official_api" | "baseline_only";
  fetchLatestModels?: (params: {
    cfg: MarvConfig;
    provider: string;
    docsUrl?: string;
  }) => Promise<RuntimeRegistryModel[]>;
};

type OpenAIModelsApiResponse = {
  data?: Array<{
    id?: string;
  }>;
};

type GoogleModelsApiResponse = {
  models?: Array<{
    name?: string;
    displayName?: string;
    inputTokenLimit?: number;
    supportedGenerationMethods?: string[];
  }>;
};

function resolveRegistryPath(): string {
  return path.join(resolveStateDir(), REGISTRY_DIRNAME, REGISTRY_FILENAME);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeTier(value: string): RuntimeRegistryModelTier {
  const lower = value.toLowerCase();
  if (lower.includes("mini") || lower.includes("flash") || lower.includes("nano")) {
    return "low";
  }
  if (
    lower.includes("pro") ||
    lower.includes("opus") ||
    lower.includes("sonnet") ||
    lower.includes("o1") ||
    lower.includes("o3") ||
    lower.includes("o4") ||
    lower.includes("gpt-5")
  ) {
    return "high";
  }
  return "standard";
}

function inferCapabilities(params: {
  provider: string;
  model: string;
  input?: Array<"text" | "image">;
  reasoning?: boolean;
}): RuntimeRegistryModelCapability[] {
  const capabilities = new Set<RuntimeRegistryModelCapability>(["text"]);
  const lower = params.model.toLowerCase();
  if (params.input?.includes("image")) {
    capabilities.add("vision");
  }
  if (
    lower.includes("vision") ||
    lower.includes("vl") ||
    lower.includes("4o") ||
    lower.includes("gemini") ||
    lower.includes("gpt-5")
  ) {
    capabilities.add("vision");
  }
  if (
    lower.includes("code") ||
    lower.includes("codex") ||
    lower.includes("coder") ||
    lower.includes("claude")
  ) {
    capabilities.add("coding");
    capabilities.add("tools");
  }
  if (params.reasoning) {
    capabilities.add("tools");
  }
  return [...capabilities];
}

function createEtag(models: RuntimeRegistryModel[]): string {
  return crypto
    .createHash("sha256")
    .update(
      JSON.stringify(
        models
          .map((entry) => ({
            ref: entry.ref,
            status: entry.status,
            source: entry.source,
          }))
          .toSorted((a, b) => a.ref.localeCompare(b.ref)),
      ),
    )
    .digest("hex");
}

function readRegistryFile(): RuntimeModelRegistry | null {
  try {
    const raw = fs.readFileSync(resolveRegistryPath(), "utf8");
    const parsed = JSON.parse(raw) as unknown;
    if (!isRecord(parsed) || !Array.isArray(parsed.models) || !isRecord(parsed.providers)) {
      return null;
    }
    return parsed as RuntimeModelRegistry;
  } catch {
    return null;
  }
}

function writeRegistryFile(registry: RuntimeModelRegistry): void {
  const registryPath = resolveRegistryPath();
  fs.mkdirSync(path.dirname(registryPath), { recursive: true });
  const tempPath = `${registryPath}.${crypto.randomUUID()}.tmp`;
  fs.writeFileSync(tempPath, `${JSON.stringify(registry, null, 2)}\n`, "utf8");
  fs.renameSync(tempPath, registryPath);
}

function normalizeCatalogEntry(entry: ModelCatalogEntry): RuntimeRegistryModel {
  const provider = normalizeProviderId(entry.provider);
  const parsed = parseModelRef(`${provider}/${entry.id}`, provider);
  const model = parsed?.model ?? entry.id.trim();
  return {
    ref: `${provider}/${model}`,
    provider,
    model,
    displayName: entry.name || entry.id,
    source: "baseline_catalog",
    status: "active",
    capabilities: inferCapabilities({
      provider,
      model,
      input: entry.input,
      reasoning: entry.reasoning,
    }),
    tier: normalizeTier(model),
    location: LOCAL_PROVIDERS.has(provider) ? "local" : "cloud",
    ...(entry.contextWindow ? { contextWindow: entry.contextWindow } : {}),
  };
}

function mergeProviderModels(params: {
  base: RuntimeRegistryModel[];
  provider: string;
  next: RuntimeRegistryModel[];
}): RuntimeRegistryModel[] {
  const filtered = params.base.filter(
    (entry) => normalizeProviderId(entry.provider) !== normalizeProviderId(params.provider),
  );
  return [...filtered, ...params.next];
}

export function listConfiguredProviders(cfg: MarvConfig, agentDir?: string): Set<string> {
  const providers = new Set<string>();
  const store = ensureAuthProfileStore(agentDir, { allowKeychainPrompt: false });
  for (const profile of Object.values(store.profiles)) {
    const provider = normalizeProviderId(String(profile.provider ?? ""));
    if (provider) {
      providers.add(provider);
    }
  }
  for (const key of Object.keys(cfg.models?.providers ?? {})) {
    const provider = normalizeProviderId(key);
    if (provider) {
      providers.add(provider);
    }
  }
  for (const provider of KNOWN_PROVIDER_GUIDES.map((entry) => entry.provider)) {
    if (
      LOCAL_PROVIDERS.has(provider) ||
      listProfilesForProvider(store, provider).length > 0 ||
      resolveEnvApiKey(provider) ||
      getCustomProviderApiKey(cfg, provider)
    ) {
      providers.add(provider);
    }
  }
  return providers;
}

async function fetchOpenAIModels(params: {
  cfg: MarvConfig;
  provider: string;
  docsUrl?: string;
}): Promise<RuntimeRegistryModel[]> {
  const resolved = await resolveApiKeyForProvider({
    provider: params.provider,
    cfg: params.cfg,
  });
  const response = await fetch("https://api.openai.com/v1/models", {
    headers: {
      Authorization: `Bearer ${resolved.apiKey}`,
    },
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) {
    throw new Error(`OpenAI models API failed: ${response.status}`);
  }
  const payload = (await response.json()) as OpenAIModelsApiResponse;
  const models = payload.data ?? [];
  return models
    .map((entry) => String(entry.id ?? "").trim())
    .filter(Boolean)
    .map((model) => ({
      ref: `${params.provider}/${model}`,
      provider: params.provider,
      model,
      displayName: model,
      source: "official_api" as const,
      status: "active" as const,
      capabilities: inferCapabilities({ provider: params.provider, model }),
      tier: normalizeTier(model),
      location: "cloud" as const,
      ...(params.docsUrl ? { docsUrl: params.docsUrl } : {}),
    }));
}

async function fetchGoogleModels(params: {
  cfg: MarvConfig;
  provider: string;
  docsUrl?: string;
}): Promise<RuntimeRegistryModel[]> {
  const resolved = await resolveApiKeyForProvider({
    provider: params.provider,
    cfg: params.cfg,
  });
  const query =
    resolved.mode === "api-key" ? `?key=${encodeURIComponent(resolved.apiKey ?? "")}` : "";
  const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models${query}`, {
    headers:
      resolved.mode === "oauth" || resolved.mode === "token"
        ? { Authorization: `Bearer ${resolved.apiKey}` }
        : undefined,
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) {
    throw new Error(`Google models API failed: ${response.status}`);
  }
  const payload = (await response.json()) as GoogleModelsApiResponse;
  const models = payload.models ?? [];
  const resolvedModels: RuntimeRegistryModel[] = [];
  for (const entry of models) {
    if (
      Array.isArray(entry.supportedGenerationMethods) &&
      !entry.supportedGenerationMethods.includes("generateContent")
    ) {
      continue;
    }
    const rawName = String(entry.name ?? "")
      .trim()
      .replace(/^models\//, "");
    if (!rawName) {
      continue;
    }
    resolvedModels.push({
      ref: `${params.provider}/${rawName}`,
      provider: params.provider,
      model: rawName,
      displayName: String(entry.displayName ?? rawName).trim() || rawName,
      source: "official_api",
      status: "active",
      capabilities: inferCapabilities({
        provider: params.provider,
        model: rawName,
        input: ["text", "image"],
      }),
      tier: normalizeTier(rawName),
      location: "cloud",
      ...(typeof entry.inputTokenLimit === "number" && entry.inputTokenLimit > 0
        ? { contextWindow: entry.inputTokenLimit }
        : {}),
      ...(params.docsUrl ? { docsUrl: params.docsUrl } : {}),
    });
  }
  return resolvedModels;
}

const KNOWN_PROVIDER_GUIDES: ProviderGuide[] = [
  {
    provider: "google",
    officialDocsUrls: ["https://ai.google.dev/gemini-api/docs/models"],
    fetchMode: "official_api",
    fetchLatestModels: fetchGoogleModels,
  },
  {
    provider: "openai",
    officialDocsUrls: ["https://platform.openai.com/docs/models"],
    fetchMode: "official_api",
    fetchLatestModels: fetchOpenAIModels,
  },
  {
    provider: "openai-codex",
    officialDocsUrls: ["https://platform.openai.com/docs/models"],
    fetchMode: "official_api",
    fetchLatestModels: fetchOpenAIModels,
  },
  {
    provider: "anthropic",
    officialDocsUrls: ["https://docs.anthropic.com/en/docs/about-claude/models/overview"],
    fetchMode: "baseline_only",
  },
];

function buildProviderStatuses(params: {
  configuredProviders: Set<string>;
  previous?: RuntimeModelRegistry | null;
}): Record<string, RuntimeRegistryProviderStatus> {
  const previousProviders = params.previous?.providers ?? {};
  return Object.fromEntries(
    KNOWN_PROVIDER_GUIDES.map((guide) => {
      const previous = previousProviders[guide.provider];
      return [
        guide.provider,
        {
          provider: guide.provider,
          officialDocsUrls: guide.officialDocsUrls,
          fetchMode: guide.fetchMode,
          configured: params.configuredProviders.has(guide.provider),
          ...(previous?.lastAttemptAt ? { lastAttemptAt: previous.lastAttemptAt } : {}),
          ...(previous?.lastSuccessAt ? { lastSuccessAt: previous.lastSuccessAt } : {}),
          ...(previous?.lastError ? { lastError: previous.lastError } : {}),
        },
      ];
    }),
  );
}

export function readRuntimeModelRegistry(): RuntimeModelRegistry | null {
  return readRegistryFile();
}

export function isRuntimeModelRegistryStale(
  registry: RuntimeModelRegistry | null,
  now = Date.now(),
): boolean {
  if (!registry?.lastSuccessfulRefreshAt) {
    return true;
  }
  return registry.lastSuccessfulRefreshAt + REGISTRY_REFRESH_INTERVAL_MS <= now;
}

export async function ensureRuntimeModelRegistry(params?: {
  cfg?: MarvConfig;
  agentDir?: string;
}): Promise<RuntimeModelRegistry> {
  const existing = readRegistryFile();
  if (existing) {
    return existing;
  }
  return await refreshRuntimeModelRegistry({
    cfg: params?.cfg,
    agentDir: params?.agentDir,
    force: true,
  });
}

export async function refreshRuntimeModelRegistry(params?: {
  cfg?: MarvConfig;
  agentDir?: string;
  force?: boolean;
}): Promise<RuntimeModelRegistry> {
  const cfg = params?.cfg;
  if (!cfg) {
    throw new Error("config required");
  }
  const now = Date.now();
  const previous = readRegistryFile();
  if (!params.force && previous && !isRuntimeModelRegistryStale(previous, now)) {
    return previous;
  }

  const catalog = await loadModelCatalog({ config: cfg, useCache: false });
  let models = catalog.map((entry) => normalizeCatalogEntry(entry));
  const configuredProviders = listConfiguredProviders(cfg, params?.agentDir);
  const providerStatuses = buildProviderStatuses({
    configuredProviders,
    previous,
  });

  for (const guide of KNOWN_PROVIDER_GUIDES) {
    const status = providerStatuses[guide.provider];
    if (!status) {
      continue;
    }
    status.lastAttemptAt = now;
    if (!status.configured || !guide.fetchLatestModels) {
      continue;
    }
    try {
      const fetched = await guide.fetchLatestModels({
        cfg,
        provider: guide.provider,
        docsUrl: guide.officialDocsUrls[0],
      });
      if (fetched.length > 0) {
        models = mergeProviderModels({
          base: models,
          provider: guide.provider,
          next: fetched,
        });
      }
      status.lastSuccessAt = now;
      delete status.lastError;
    } catch (error) {
      status.lastError = error instanceof Error ? error.message : String(error);
    }
  }

  const deduped = Array.from(
    new Map(
      models
        .filter((entry) => entry.status !== "removed")
        .map((entry) => [entry.ref, entry] as const),
    ).values(),
  ).toSorted((a, b) => a.ref.localeCompare(b.ref));

  const registry: RuntimeModelRegistry = {
    version: REGISTRY_VERSION,
    generatedAt: now,
    lastSuccessfulRefreshAt: now,
    nextRefreshAfter: now + REGISTRY_REFRESH_INTERVAL_MS,
    providers: providerStatuses,
    models: deduped,
    etag: createEtag(deduped),
  };
  writeRegistryFile(registry);
  return registry;
}

export function listRuntimeRegistryModelsForProviders(params: {
  registry: RuntimeModelRegistry | null;
  providers: Set<string>;
}): RuntimeRegistryModel[] {
  if (!params.registry) {
    return [];
  }
  return params.registry.models.filter((entry) =>
    params.providers.has(normalizeProviderId(entry.provider)),
  );
}

export function resolveRuntimeRegistryPathForDisplay(): string {
  return resolveRegistryPath();
}

export function startRuntimeModelRegistryRefreshLoop(params: {
  cfg: MarvConfig;
  agentDir?: string;
  log?: { warn: (message: string) => void };
}): void {
  if (refreshTimer) {
    return;
  }
  const runRefresh = () => {
    void refreshRuntimeModelRegistry({
      cfg: params.cfg,
      agentDir: params.agentDir,
      force: false,
    }).catch((error) => {
      params.log?.warn?.(`[model-registry] refresh failed: ${String(error)}`);
    });
  };
  runRefresh();
  refreshTimer = setInterval(runRefresh, REGISTRY_CHECK_INTERVAL_MS);
  refreshTimer.unref?.();
}

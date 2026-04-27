import type { MarvConfig } from "../../core/config/config.js";
import {
  ensureAuthProfileStore,
  getSoonestCooldownExpiry,
  isProfileInCooldown,
  resolveAuthProfileOrder,
} from "../auth-profiles.js";
import { DEFAULT_MODEL, DEFAULT_PROVIDER } from "../defaults.js";
import {
  coerceToFailoverError,
  describeFailoverError,
  isFailoverError,
  isTimeoutError,
} from "../failover-error.js";
import type { FailoverReason } from "../runner/pi-embedded-helpers.js";
import { isLikelyContextOverflowError } from "../runner/pi-embedded-helpers.js";
import { markRuntimeModelFailure, markRuntimeModelReady } from "./model-availability-state.js";
import { isRuntimeLocalProvider, resolveRuntimeModelPlan } from "./model-pool.js";
import {
  buildConfiguredSelectionKeys,
  modelKey,
  normalizeModelRef,
  normalizeProviderId,
  parseModelRef,
  resolveConfiguredModelRef,
} from "./model-resolve.js";
import { resolveOrderedModelRoutePlan } from "./model-route.js";

type ModelCandidate = {
  provider: string;
  model: string;
};

export type FallbackAttempt = {
  provider: string;
  model: string;
  error: string;
  reason?: FailoverReason;
  status?: number;
  code?: string;
  local?: boolean;
};

/**
 * Fallback abort check. Only treats explicit AbortError names as user aborts.
 * Message-based checks (e.g., "aborted") can mask timeouts and skip fallback.
 */
function isFallbackAbortError(err: unknown): boolean {
  if (!err || typeof err !== "object") {
    return false;
  }
  if (isFailoverError(err)) {
    return false;
  }
  const name = "name" in err ? String(err.name) : "";
  return name === "AbortError";
}

function shouldRethrowAbort(err: unknown): boolean {
  return isFallbackAbortError(err) && !isTimeoutError(err);
}

function createModelCandidateCollector(allowlist: Set<string> | null | undefined): {
  candidates: ModelCandidate[];
  addCandidate: (candidate: ModelCandidate, enforceAllowlist: boolean) => void;
} {
  const seen = new Set<string>();
  const candidates: ModelCandidate[] = [];

  const addCandidate = (candidate: ModelCandidate, enforceAllowlist: boolean) => {
    if (!candidate.provider || !candidate.model) {
      return;
    }
    const key = modelKey(candidate.provider, candidate.model);
    if (seen.has(key)) {
      return;
    }
    if (enforceAllowlist && allowlist && !allowlist.has(key)) {
      return;
    }
    seen.add(key);
    candidates.push(candidate);
  };

  return { candidates, addCandidate };
}

type ModelFallbackErrorHandler = (attempt: {
  provider: string;
  model: string;
  error: unknown;
  attempt: number;
  total: number;
}) => void | Promise<void>;

type ModelFallbackRunResult<T> = {
  result: T;
  provider: string;
  model: string;
  attempts: FallbackAttempt[];
  notice?: string;
};

function formatLocalModelFallbackNotice(params: {
  attempts: FallbackAttempt[];
  provider: string;
  model: string;
}): string | undefined {
  const localRefs = [
    ...new Set(
      params.attempts
        .filter((attempt) => attempt.local)
        .map((attempt) => `${attempt.provider}/${attempt.model}`),
    ),
  ];
  if (localRefs.length === 0) {
    return undefined;
  }
  return `Local model ${localRefs.join(", ")} is unavailable; temporarily used ${params.provider}/${params.model}. Start the local model server and Marv will try the configured order again next run.`;
}

function throwFallbackFailureSummary(params: {
  attempts: FallbackAttempt[];
  candidates: ModelCandidate[];
  lastError: unknown;
  label: string;
  formatAttempt: (attempt: FallbackAttempt) => string;
}): never {
  if (params.attempts.length <= 1 && params.lastError) {
    throw params.lastError;
  }
  const summary =
    params.attempts.length > 0 ? params.attempts.map(params.formatAttempt).join(" | ") : "unknown";
  throw new Error(
    `All ${params.label} failed (${params.attempts.length || params.candidates.length}): ${summary}`,
    {
      cause: params.lastError instanceof Error ? params.lastError : undefined,
    },
  );
}

function resolveImageFallbackCandidates(params: {
  cfg: MarvConfig | undefined;
  defaultProvider: string;
  modelOverride?: string;
  agentDir?: string;
}): ModelCandidate[] {
  const plan = params.cfg
    ? resolveRuntimeModelPlan({
        cfg: params.cfg,
        agentDir: params.agentDir,
        requirements: { requiredCapabilities: ["text", "vision"] },
      })
    : null;
  const candidates: ModelCandidate[] = [];
  const push = (provider: string, model: string) => {
    if (!candidates.some((entry) => entry.provider === provider && entry.model === model)) {
      candidates.push({ provider, model });
    }
  };
  if (params.modelOverride?.trim()) {
    const override = normalizeModelRef(params.defaultProvider, params.modelOverride);
    push(override.provider, override.model);
  }
  for (const entry of plan?.candidates ?? []) {
    push(entry.provider, entry.model);
  }
  return candidates;
}

function resolveFallbackCandidates(params: {
  cfg: MarvConfig | undefined;
  provider: string;
  model: string;
  agentId?: string;
  /** Optional explicit fallback refs; when provided, uses this ordered list after the primary. */
  fallbacksOverride?: string[];
}): ModelCandidate[] {
  const primary = params.cfg
    ? resolveConfiguredModelRef({
        cfg: params.cfg,
        defaultProvider: DEFAULT_PROVIDER,
        defaultModel: DEFAULT_MODEL,
      })
    : null;
  const defaultProvider = primary?.provider ?? DEFAULT_PROVIDER;
  const defaultModel = primary?.model ?? DEFAULT_MODEL;
  const providerRaw = String(params.provider ?? "").trim() || defaultProvider;
  const modelRaw = String(params.model ?? "").trim() || defaultModel;
  const normalizedPrimary = normalizeModelRef(providerRaw, modelRaw);
  const allowlist = buildConfiguredSelectionKeys({
    cfg: params.cfg,
    defaultProvider,
  });
  const { candidates, addCandidate } = createModelCandidateCollector(allowlist);

  addCandidate(normalizedPrimary, false);
  if (params.fallbacksOverride !== undefined) {
    for (const raw of params.fallbacksOverride) {
      const resolved = parseModelRef(String(raw ?? ""), defaultProvider);
      if (resolved) {
        addCandidate(resolved, false);
      }
    }
    return candidates;
  }

  const configuredRoute = resolveOrderedModelRoutePlan({
    cfg: params.cfg,
    agentId: params.agentId,
    primary: normalizedPrimary,
    defaultProvider,
  });
  if (configuredRoute.hasConfiguredRoute) {
    for (const entry of configuredRoute.entries.slice(1)) {
      addCandidate({ provider: entry.provider, model: entry.model }, false);
    }
    return candidates;
  }

  const runtimePlan = params.cfg
    ? resolveRuntimeModelPlan({
        cfg: params.cfg,
        agentDir: undefined,
        requirements: { requiredCapabilities: ["text"] },
      })
    : null;
  for (const entry of runtimePlan?.candidates ?? []) {
    addCandidate({ provider: entry.provider, model: entry.model }, true);
  }

  return candidates;
}

const lastProbeAttempt = new Map<string, number>();
const MIN_PROBE_INTERVAL_MS = 30_000; // 30 seconds between probes per key
const PROBE_MARGIN_MS = 2 * 60 * 1000;
const PROBE_SCOPE_DELIMITER = "::";

function resolveProbeThrottleKey(provider: string, agentDir?: string): string {
  const scope = String(agentDir ?? "").trim();
  return scope ? `${scope}${PROBE_SCOPE_DELIMITER}${provider}` : provider;
}

function shouldProbePrimaryDuringCooldown(params: {
  isPrimary: boolean;
  hasFallbackCandidates: boolean;
  now: number;
  throttleKey: string;
  authStore: ReturnType<typeof ensureAuthProfileStore>;
  profileIds: string[];
}): boolean {
  if (!params.isPrimary || !params.hasFallbackCandidates) {
    return false;
  }

  const lastProbe = lastProbeAttempt.get(params.throttleKey) ?? 0;
  if (params.now - lastProbe < MIN_PROBE_INTERVAL_MS) {
    return false;
  }

  const soonest = getSoonestCooldownExpiry(params.authStore, params.profileIds);
  if (soonest === null || !Number.isFinite(soonest)) {
    return true;
  }

  // Probe when cooldown already expired or within the configured margin.
  return params.now >= soonest - PROBE_MARGIN_MS;
}

/** @internal – exposed for unit tests only */
export const _probeThrottleInternals = {
  lastProbeAttempt,
  MIN_PROBE_INTERVAL_MS,
  PROBE_MARGIN_MS,
  resolveProbeThrottleKey,
} as const;

export async function runWithModelFallback<T>(params: {
  cfg: MarvConfig | undefined;
  provider: string;
  model: string;
  agentDir?: string;
  agentId?: string;
  /** Optional explicit fallback refs; when provided, uses this ordered list after the primary. */
  fallbacksOverride?: string[];
  run: (provider: string, model: string) => Promise<T>;
  onError?: ModelFallbackErrorHandler;
}): Promise<ModelFallbackRunResult<T>> {
  const candidates = resolveFallbackCandidates({
    cfg: params.cfg,
    provider: params.provider,
    model: params.model,
    agentId: params.agentId,
    fallbacksOverride: params.fallbacksOverride,
  });
  const authStore = params.cfg
    ? ensureAuthProfileStore(params.agentDir, { allowKeychainPrompt: false })
    : null;
  const attempts: FallbackAttempt[] = [];
  let lastError: unknown;

  const hasFallbackCandidates = candidates.length > 1;

  // Track individual models that returned 429 during this run.
  // Different models under the same provider may have different rate limits,
  // so we try each model individually. Only skip remaining models from a
  // provider when ALL its candidates have been individually rate-limited.
  const rateLimitedModels = new Set<string>();

  // Pre-compute per-provider model counts for provider-level skip detection.
  const modelsPerProvider = new Map<string, number>();
  for (const c of candidates) {
    const pk = normalizeProviderId(c.provider);
    modelsPerProvider.set(pk, (modelsPerProvider.get(pk) ?? 0) + 1);
  }

  // Count rate-limited models per provider during this run.
  const rateLimitedCountPerProvider = new Map<string, number>();

  function isProviderFullyRateLimited(providerKey: string): boolean {
    const total = modelsPerProvider.get(providerKey) ?? 0;
    const limited = rateLimitedCountPerProvider.get(providerKey) ?? 0;
    return total > 0 && limited >= total;
  }

  for (let i = 0; i < candidates.length; i += 1) {
    const candidate = candidates[i];
    const candidateProviderKey = normalizeProviderId(candidate.provider);
    const isLocalCandidate = isRuntimeLocalProvider(candidate.provider, params.cfg);

    // Skip if ALL models from this provider have been rate-limited in this run.
    if (isProviderFullyRateLimited(candidateProviderKey)) {
      attempts.push({
        provider: candidate.provider,
        model: candidate.model,
        error: `Provider ${candidate.provider} fully rate-limited (all models exhausted)`,
        reason: "rate_limit",
        local: isLocalCandidate,
      });
      continue;
    }

    if (authStore) {
      const profileIds = resolveAuthProfileOrder({
        cfg: params.cfg,
        store: authStore,
        provider: candidate.provider,
      });
      const isAnyProfileAvailable = profileIds.some((id) => !isProfileInCooldown(authStore, id));

      if (profileIds.length > 0 && !isAnyProfileAvailable) {
        // All profiles for this provider are in cooldown.
        // For the primary model (i === 0), probe it if the soonest cooldown
        // expiry is close or already past. This avoids staying on a fallback
        // model long after the real rate-limit window clears.
        const now = Date.now();
        const probeThrottleKey = resolveProbeThrottleKey(candidate.provider, params.agentDir);
        const shouldProbe = shouldProbePrimaryDuringCooldown({
          isPrimary: i === 0,
          hasFallbackCandidates,
          now,
          throttleKey: probeThrottleKey,
          authStore,
          profileIds,
        });
        if (!shouldProbe) {
          attempts.push({
            provider: candidate.provider,
            model: candidate.model,
            error: `Provider ${candidate.provider} is in cooldown (all profiles unavailable)`,
            reason: "rate_limit",
            local: isLocalCandidate,
          });
          continue;
        }
        lastProbeAttempt.set(probeThrottleKey, now);
      }
    }
    try {
      const result = await params.run(candidate.provider, candidate.model);
      markRuntimeModelReady(modelKey(candidate.provider, candidate.model));
      const notice = formatLocalModelFallbackNotice({
        attempts,
        provider: candidate.provider,
        model: candidate.model,
      });
      return {
        result,
        provider: candidate.provider,
        model: candidate.model,
        attempts,
        ...(notice ? { notice } : {}),
      };
    } catch (err) {
      if (shouldRethrowAbort(err)) {
        throw err;
      }
      const errMessage = err instanceof Error ? err.message : String(err);
      if (isLikelyContextOverflowError(errMessage)) {
        throw err;
      }
      const normalized =
        coerceToFailoverError(err, {
          provider: candidate.provider,
          model: candidate.model,
        }) ?? err;
      if (!isFailoverError(normalized)) {
        throw err;
      }

      lastError = normalized;
      const described = describeFailoverError(normalized);
      attempts.push({
        provider: candidate.provider,
        model: candidate.model,
        error: described.message,
        reason: described.reason,
        status: described.status,
        code: described.code,
        local: isLocalCandidate,
      });
      markRuntimeModelFailure({
        ref: modelKey(candidate.provider, candidate.model),
        error: normalized,
        persist: !isLocalCandidate,
      });

      // Per-model rate-limit tracking: mark this specific model as rate-limited.
      // The provider is only skipped when ALL its candidate models are rate-limited.
      if (described.reason === "rate_limit") {
        const mk = modelKey(candidate.provider, candidate.model);
        if (!rateLimitedModels.has(mk)) {
          rateLimitedModels.add(mk);
          rateLimitedCountPerProvider.set(
            candidateProviderKey,
            (rateLimitedCountPerProvider.get(candidateProviderKey) ?? 0) + 1,
          );
        }
      }

      await params.onError?.({
        provider: candidate.provider,
        model: candidate.model,
        error: normalized,
        attempt: i + 1,
        total: candidates.length,
      });
    }
  }

  throwFallbackFailureSummary({
    attempts,
    candidates,
    lastError,
    label: "models",
    formatAttempt: (attempt) =>
      `${attempt.provider}/${attempt.model}: ${attempt.error}${
        attempt.reason ? ` (${attempt.reason})` : ""
      }`,
  });
}

export async function runWithImageModelFallback<T>(params: {
  cfg: MarvConfig | undefined;
  modelOverride?: string;
  run: (provider: string, model: string) => Promise<T>;
  onError?: ModelFallbackErrorHandler;
}): Promise<ModelFallbackRunResult<T>> {
  const candidates = resolveImageFallbackCandidates({
    cfg: params.cfg,
    defaultProvider: DEFAULT_PROVIDER,
    modelOverride: params.modelOverride,
  });
  if (candidates.length === 0) {
    throw new Error("No image-capable model is available in the configured model pool.");
  }

  const attempts: FallbackAttempt[] = [];
  let lastError: unknown;

  for (let i = 0; i < candidates.length; i += 1) {
    const candidate = candidates[i];
    try {
      const result = await params.run(candidate.provider, candidate.model);
      markRuntimeModelReady(modelKey(candidate.provider, candidate.model));
      return {
        result,
        provider: candidate.provider,
        model: candidate.model,
        attempts,
      };
    } catch (err) {
      if (shouldRethrowAbort(err)) {
        throw err;
      }
      lastError = err;
      attempts.push({
        provider: candidate.provider,
        model: candidate.model,
        error: err instanceof Error ? err.message : String(err),
      });
      markRuntimeModelFailure({
        ref: modelKey(candidate.provider, candidate.model),
        error: err,
      });
      await params.onError?.({
        provider: candidate.provider,
        model: candidate.model,
        error: err,
        attempt: i + 1,
        total: candidates.length,
      });
    }
  }

  throwFallbackFailureSummary({
    attempts,
    candidates,
    lastError,
    label: "image models",
    formatAttempt: (attempt) => `${attempt.provider}/${attempt.model}: ${attempt.error}`,
  });
}

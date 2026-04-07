import { getCustomProviderApiKey, resolveEnvApiKey } from "../../agents/model/model-auth.js";
import {
  clearAllRuntimeModelAvailability,
  clearRuntimeModelAvailability,
  markRuntimeModelFailure,
  markRuntimeModelReady,
  readRuntimeModelAvailability,
  type RuntimeModelAvailabilityEntry,
} from "../../agents/model/model-availability-state.js";
import { resolveRuntimeModelPlan } from "../../agents/model/model-pool.js";
import { normalizeProviderId } from "../../agents/model/model-selection.js";
import { loadConfig, type MarvConfig } from "../../core/config/config.js";
import { fetchWithPrivateNetworkAccess } from "../../infra/net/private-network-fetch.js";
import type { RuntimeEnv } from "../../runtime.js";

// Pseudo-status applied to configured refs that have no cached availability
// entry yet (no run/probe has marked them ready or failed). Without this,
// configured refs that genuinely exist in provider inventory silently
// disappear from `pool list`, making the command look like an authoritative
// but sparse inventory.
type PoolListStatus = RuntimeModelAvailabilityEntry["status"] | "unprobed";

type PoolListRow = {
  ref: string;
  status: PoolListStatus;
  lastCheckedAt?: number;
  lastError?: string;
  retryAfter?: number;
  configured: boolean;
};

function buildPoolRows(): PoolListRow[] {
  const store = readRuntimeModelAvailability();
  const rows = new Map<string, PoolListRow>();

  for (const [ref, entry] of Object.entries(store.models)) {
    rows.set(ref, {
      ref,
      status: entry.status,
      lastCheckedAt: entry.lastCheckedAt,
      lastError: entry.lastError,
      retryAfter: entry.retryAfter,
      configured: false,
    });
  }

  // Merge configured refs so operators can see models that are in config but
  // have not yet been probed by a run. Safe-guard against config load errors
  // so `pool list` still works even if the config is broken.
  const orderedRefs: string[] = [];
  try {
    const cfg = loadConfig();
    const plan = resolveRuntimeModelPlan({ cfg });
    for (const configured of plan.configured) {
      orderedRefs.push(configured.ref);
      const existing = rows.get(configured.ref);
      if (existing) {
        existing.configured = true;
        continue;
      }
      rows.set(configured.ref, {
        ref: configured.ref,
        status: "unprobed",
        configured: true,
      });
    }
  } catch {
    // Config unavailable / invalid: fall back to raw store view.
  }

  const remaining = [...rows.values()].filter((row) => !orderedRefs.includes(row.ref));
  return [
    ...orderedRefs.map((ref) => rows.get(ref)).filter((row): row is PoolListRow => Boolean(row)),
    ...remaining.toSorted((a, b) => a.ref.localeCompare(b.ref)),
  ];
}

export async function modelsPoolListCommand(
  opts: { json?: boolean; plain?: boolean },
  runtime: RuntimeEnv,
) {
  const rows = buildPoolRows();

  if (opts.json) {
    runtime.log(
      JSON.stringify(
        {
          models: Object.fromEntries(
            rows.map((row) => [
              row.ref,
              {
                status: row.status,
                lastCheckedAt: row.lastCheckedAt,
                lastError: row.lastError,
                retryAfter: row.retryAfter,
                configured: row.configured,
              },
            ]),
          ),
        },
        null,
        2,
      ),
    );
    return;
  }

  if (rows.length === 0) {
    if (!opts.plain) {
      runtime.log("No configured models and no availability entries.");
    }
    return;
  }

  if (opts.plain) {
    for (const row of rows) {
      const parts = [row.ref, row.status];
      if (row.lastError) {
        parts.push(row.lastError);
      }
      runtime.log(parts.join("\t"));
    }
    return;
  }

  runtime.log(`Model availability (${rows.length}):\n`);
  for (const row of rows) {
    const age = row.lastCheckedAt ? formatAge(Date.now() - row.lastCheckedAt) : "never";
    const retryIn =
      row.retryAfter && row.retryAfter > Date.now()
        ? `, retry in ${formatAge(row.retryAfter - Date.now())}`
        : "";
    const configuredTag = row.configured ? " [configured]" : "";
    runtime.log(`  ${row.ref}${configuredTag}`);
    runtime.log(`    status: ${row.status}${retryIn}`);
    runtime.log(`    checked: ${age === "never" ? "never" : `${age} ago`}`);
    if (row.lastError) {
      runtime.log(`    error: ${row.lastError}`);
    }
    runtime.log("");
  }
}

export async function modelsPoolClearCommand(modelRef: string | undefined, runtime: RuntimeEnv) {
  if (modelRef) {
    const cleared = clearRuntimeModelAvailability(modelRef);
    if (cleared) {
      runtime.log(`Cleared availability state for ${modelRef}.`);
    } else {
      runtime.log(`No availability entry found for ${modelRef}.`);
    }
    return;
  }

  const count = clearAllRuntimeModelAvailability();
  if (count > 0) {
    runtime.log(`Cleared ${count} model availability ${count === 1 ? "entry" : "entries"}.`);
  } else {
    runtime.log("No model availability entries to clear.");
  }
}

type ProviderProbeOutcome =
  | { kind: "ok"; models: Set<string> }
  | { kind: "http_error"; message: string }
  | { kind: "no_base_url" }
  | { kind: "fetch_failed"; message: string };

async function probeProviderModels(params: {
  cfg: MarvConfig;
  provider: string;
  baseUrl: string;
  timeoutMs: number;
}): Promise<ProviderProbeOutcome> {
  // Most OpenAI-compatible servers expose model inventory at `/v1/models`.
  // Normalize both `.../v1` and `...` forms so we don't double the suffix.
  const trimmed = params.baseUrl.replace(/\/+$/, "");
  const url = trimmed.replace(/\/v1$/i, "") + "/v1/models";
  const apiKey =
    getCustomProviderApiKey(params.cfg, params.provider) ??
    resolveEnvApiKey(params.provider)?.apiKey;
  const headers: Record<string, string> = { Accept: "application/json" };
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }
  let release: (() => Promise<void>) | undefined;
  try {
    const guarded = await fetchWithPrivateNetworkAccess({
      url,
      timeoutMs: params.timeoutMs,
      auditContext: `pool-refresh.${params.provider}`,
      init: { headers },
    });
    release = guarded.release;
    if (!guarded.response.ok) {
      return { kind: "http_error", message: `HTTP ${guarded.response.status}` };
    }
    const payload = (await guarded.response.json()) as {
      data?: Array<{ id?: string }>;
    };
    const models = new Set(
      (payload.data ?? [])
        .map((entry) => String(entry?.id ?? "").trim())
        .filter((id): id is string => Boolean(id)),
    );
    return { kind: "ok", models };
  } catch (error) {
    return {
      kind: "fetch_failed",
      message: error instanceof Error ? error.message : String(error),
    };
  } finally {
    if (release) {
      await release();
    }
  }
}

export async function modelsPoolRefreshCommand(
  opts: { json?: boolean; timeout?: string },
  runtime: RuntimeEnv,
) {
  const timeoutMs = opts.timeout ? Number(opts.timeout) : 5000;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error("--timeout must be a positive number (ms).");
  }

  const cfg = loadConfig();
  const plan = resolveRuntimeModelPlan({ cfg });

  // Group configured refs by provider so we probe each provider once.
  const byProvider = new Map<string, { models: string[]; baseUrl?: string }>();
  const providerConfigs = cfg.models?.providers ?? {};
  for (const configured of plan.configured) {
    const norm = normalizeProviderId(configured.provider);
    const group = byProvider.get(norm) ?? { models: [] };
    group.models.push(configured.model);
    if (!group.baseUrl) {
      // Find the matching provider entry (key may differ in case/format).
      for (const [key, entry] of Object.entries(providerConfigs)) {
        if (normalizeProviderId(key) === norm && entry?.baseUrl) {
          group.baseUrl = entry.baseUrl;
          break;
        }
      }
      // Ollama has a well-known default even without explicit config.
      if (!group.baseUrl && norm === "ollama") {
        group.baseUrl = "http://127.0.0.1:11434";
      }
    }
    byProvider.set(norm, group);
  }

  type RefreshRow = {
    ref: string;
    provider: string;
    model: string;
    outcome: "ready" | "unsupported" | "provider_unreachable" | "skipped";
    detail?: string;
  };
  const rows: RefreshRow[] = [];

  for (const [provider, group] of byProvider) {
    if (!group.baseUrl) {
      for (const model of group.models) {
        rows.push({
          ref: `${provider}/${model}`,
          provider,
          model,
          outcome: "skipped",
          detail: "no baseUrl configured",
        });
      }
      continue;
    }
    const probe = await probeProviderModels({
      cfg,
      provider,
      baseUrl: group.baseUrl,
      timeoutMs,
    });
    for (const model of group.models) {
      const ref = `${provider}/${model}`;
      if (probe.kind === "ok") {
        if (probe.models.has(model)) {
          markRuntimeModelReady(ref);
          rows.push({ ref, provider, model, outcome: "ready" });
        } else {
          // Provider responded but doesn't have this model. Surface as a
          // concrete drift signal via markRuntimeModelFailure (maps to
          // `unsupported` for a "model not found" error).
          markRuntimeModelFailure({
            ref,
            error: new Error("model not found in provider inventory"),
          });
          rows.push({
            ref,
            provider,
            model,
            outcome: "unsupported",
            detail: "not in /v1/models",
          });
        }
      } else if (probe.kind === "http_error" || probe.kind === "fetch_failed") {
        // Don't overwrite existing availability state; we can't distinguish
        // "provider down" from "provider misconfigured" reliably.
        rows.push({
          ref,
          provider,
          model,
          outcome: "provider_unreachable",
          detail: probe.message,
        });
      } else {
        rows.push({
          ref,
          provider,
          model,
          outcome: "skipped",
          detail: "no baseUrl configured",
        });
      }
    }
  }

  if (opts.json) {
    runtime.log(JSON.stringify({ refreshed: rows }, null, 2));
    return;
  }

  const counts = {
    ready: 0,
    unsupported: 0,
    provider_unreachable: 0,
    skipped: 0,
  };
  for (const row of rows) {
    counts[row.outcome] += 1;
  }
  runtime.log(
    `Refreshed ${rows.length} configured model ref(s): ` +
      `${counts.ready} ready, ${counts.unsupported} unsupported, ` +
      `${counts.provider_unreachable} unreachable, ${counts.skipped} skipped.\n`,
  );
  for (const row of rows.toSorted((a, b) => a.ref.localeCompare(b.ref))) {
    const detail = row.detail ? ` (${row.detail})` : "";
    runtime.log(`  ${row.ref}: ${row.outcome}${detail}`);
  }
}

function formatAge(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes > 0 ? `${hours}h${remainingMinutes}m` : `${hours}h`;
}

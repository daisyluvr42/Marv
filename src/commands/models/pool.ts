import {
  clearAllRuntimeModelAvailability,
  clearRuntimeModelAvailability,
  readRuntimeModelAvailability,
  type RuntimeModelAvailabilityEntry,
} from "../../agents/model/model-availability-state.js";
import { resolveRuntimeModelPlan } from "../../agents/model/model-pool.js";
import { loadConfig } from "../../core/config/config.js";
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
  try {
    const cfg = loadConfig();
    const plan = resolveRuntimeModelPlan({ cfg });
    for (const configured of plan.configured) {
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

  return [...rows.values()].toSorted((a, b) => a.ref.localeCompare(b.ref));
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

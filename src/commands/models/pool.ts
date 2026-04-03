import {
  clearAllRuntimeModelAvailability,
  clearRuntimeModelAvailability,
  readRuntimeModelAvailability,
} from "../../agents/model/model-availability-state.js";
import type { RuntimeEnv } from "../../runtime.js";

export async function modelsPoolListCommand(
  opts: { json?: boolean; plain?: boolean },
  runtime: RuntimeEnv,
) {
  const store = readRuntimeModelAvailability();
  const entries = Object.entries(store.models);

  if (opts.json) {
    runtime.log(JSON.stringify(store, null, 2));
    return;
  }

  if (entries.length === 0) {
    if (!opts.plain) {
      runtime.log("No model availability entries.");
    }
    return;
  }

  if (opts.plain) {
    for (const [ref, entry] of entries) {
      const parts = [ref, entry.status];
      if (entry.lastError) {
        parts.push(entry.lastError);
      }
      runtime.log(parts.join("\t"));
    }
    return;
  }

  runtime.log(`Model availability (${entries.length}):\n`);
  for (const [ref, entry] of entries) {
    const age = entry.lastCheckedAt ? formatAge(Date.now() - entry.lastCheckedAt) : "unknown";
    const retryIn =
      entry.retryAfter && entry.retryAfter > Date.now()
        ? `, retry in ${formatAge(entry.retryAfter - Date.now())}`
        : "";
    runtime.log(`  ${ref}`);
    runtime.log(`    status: ${entry.status}${retryIn}`);
    runtime.log(`    checked: ${age} ago`);
    if (entry.lastError) {
      runtime.log(`    error: ${entry.lastError}`);
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

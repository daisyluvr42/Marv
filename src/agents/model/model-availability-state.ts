import fs from "node:fs";
import path from "node:path";
import { resolveStateDir } from "../../core/config/paths.js";
import { describeFailoverError } from "../failover-error.js";

export type RuntimeModelAvailabilityStatus =
  | "ready"
  | "temporary_unavailable"
  | "unsupported"
  | "auth_invalid";

export type RuntimeModelAvailabilityEntry = {
  status: RuntimeModelAvailabilityStatus;
  lastCheckedAt: number;
  lastError?: string;
  retryAfter?: number;
};

type RuntimeModelAvailabilityStore = {
  models: Record<string, RuntimeModelAvailabilityEntry>;
};

const AVAILABILITY_FILENAME = "model-availability.json";
const TEMPORARY_UNAVAILABLE_RETRY_MS = {
  rate_limit: 15 * 60 * 1000,
  timeout: 5 * 60 * 1000,
  billing: 60 * 60 * 1000,
} as const;

/** Unsupported/auth_invalid entries expire after 30 minutes so local models can recover. */
const PERSISTENT_STATUS_TTL_MS = 30 * 60 * 1000;

function resolveAvailabilityPath(): string {
  return path.join(resolveStateDir(), AVAILABILITY_FILENAME);
}

function readStore(): RuntimeModelAvailabilityStore {
  const filePath = resolveAvailabilityPath();
  try {
    const raw = fs.readFileSync(filePath, "utf8");
    const parsed = JSON.parse(raw) as RuntimeModelAvailabilityStore;
    return parsed && typeof parsed === "object" && parsed.models ? parsed : { models: {} };
  } catch {
    return { models: {} };
  }
}

function writeStore(store: RuntimeModelAvailabilityStore): void {
  const filePath = resolveAvailabilityPath();
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(store, null, 2)}\n`, "utf8");
}

function isUnsupportedMessage(message: string): boolean {
  const lower = message.toLowerCase();
  return (
    lower.includes("unknown model") ||
    lower.includes("model does not exist") ||
    lower.includes("model not found") ||
    lower.includes("invalid model") ||
    lower.includes("not supported") ||
    lower.includes("unsupported model") ||
    lower.includes("model has been deprecated") ||
    lower.includes("deprecated and will be removed") ||
    lower.includes("no access to model") ||
    lower.includes("do not have access to model") ||
    lower.includes("not authorized to access this model") ||
    lower.includes("not available for your account") ||
    lower.includes("does not have access to model") ||
    lower.includes("access to the requested model is denied")
  );
}

function resolveRetryAfter(params: {
  status: RuntimeModelAvailabilityStatus;
  reason?: ReturnType<typeof describeFailoverError>["reason"];
  now: number;
}): number | undefined {
  if (params.status !== "temporary_unavailable" || !params.reason) {
    return undefined;
  }
  switch (params.reason) {
    case "rate_limit":
      return params.now + TEMPORARY_UNAVAILABLE_RETRY_MS.rate_limit;
    case "timeout":
      return params.now + TEMPORARY_UNAVAILABLE_RETRY_MS.timeout;
    case "billing":
      return params.now + TEMPORARY_UNAVAILABLE_RETRY_MS.billing;
    default:
      return undefined;
  }
}

export function readRuntimeModelAvailability(): RuntimeModelAvailabilityStore {
  return readStore();
}

export function getRuntimeModelAvailability(
  ref: string,
): RuntimeModelAvailabilityEntry | undefined {
  const store = readStore();
  const entry = store.models[ref];
  if (!entry) {
    return undefined;
  }
  const now = Date.now();
  if (
    entry.status === "temporary_unavailable" &&
    typeof entry.retryAfter === "number" &&
    entry.retryAfter <= now
  ) {
    delete store.models[ref];
    writeStore(store);
    return undefined;
  }
  // Expire persistent failure states (unsupported/auth_invalid) after TTL
  // so local/custom models that were transiently unavailable can recover.
  if (
    (entry.status === "unsupported" || entry.status === "auth_invalid") &&
    typeof entry.lastCheckedAt === "number" &&
    now - entry.lastCheckedAt >= PERSISTENT_STATUS_TTL_MS
  ) {
    delete store.models[ref];
    writeStore(store);
    return undefined;
  }
  return entry;
}

/**
 * Clear all unsupported/auth_invalid failure states for models belonging to
 * a provider. Called when the user reconfigures auth for a provider so
 * previously rejected models can re-enter the candidate pool.
 */
export function clearProviderFailureStates(provider: string): void {
  const store = readStore();
  const normalizedProvider = provider.toLowerCase().trim();
  let changed = false;
  for (const [ref, entry] of Object.entries(store.models)) {
    if (entry.status !== "unsupported" && entry.status !== "auth_invalid") {
      continue;
    }
    // ref format: "provider/model"
    const slash = ref.indexOf("/");
    const refProvider = slash > 0 ? ref.slice(0, slash).toLowerCase().trim() : "";
    if (refProvider === normalizedProvider) {
      delete store.models[ref];
      changed = true;
    }
  }
  if (changed) {
    writeStore(store);
  }
}

/**
 * Clear the availability entry for a single model ref so it re-enters
 * the candidate pool immediately.
 */
export function clearRuntimeModelAvailability(ref: string): boolean {
  const store = readStore();
  if (!store.models[ref]) {
    return false;
  }
  delete store.models[ref];
  writeStore(store);
  return true;
}

/**
 * Clear all entries from the availability store (full reset).
 * Returns the number of entries removed.
 */
export function clearAllRuntimeModelAvailability(): number {
  const store = readStore();
  const count = Object.keys(store.models).length;
  if (count === 0) {
    return 0;
  }
  writeStore({ models: {} });
  return count;
}

export function markRuntimeModelReady(ref: string): void {
  const store = readStore();
  store.models[ref] = {
    status: "ready",
    lastCheckedAt: Date.now(),
  };
  writeStore(store);
}

export function markRuntimeModelFailure(params: {
  ref: string;
  error: unknown;
  persist?: boolean;
}): RuntimeModelAvailabilityStatus {
  const described = describeFailoverError(params.error);
  const message = described.message.trim();
  const now = Date.now();
  let status: RuntimeModelAvailabilityStatus = "temporary_unavailable";

  if (isUnsupportedMessage(message) || described.status === 404 || described.status === 410) {
    status = "unsupported";
  } else if (described.reason === "auth") {
    status = "auth_invalid";
  }
  const retryAfter = resolveRetryAfter({ status, reason: described.reason, now });

  if (params.persist === false) {
    return status;
  }

  const store = readStore();
  store.models[params.ref] = {
    status,
    lastCheckedAt: now,
    ...(message ? { lastError: message } : {}),
    ...(retryAfter ? { retryAfter } : {}),
  };
  writeStore(store);
  return status;
}

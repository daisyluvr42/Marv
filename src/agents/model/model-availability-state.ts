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
    lower.includes("model not found") ||
    lower.includes("invalid model") ||
    lower.includes("not supported") ||
    lower.includes("unsupported model") ||
    lower.includes("model has been deprecated") ||
    lower.includes("no access to model") ||
    lower.includes("not available for your account") ||
    lower.includes("does not have access to model") ||
    lower.includes("context window too small")
  );
}

export function readRuntimeModelAvailability(): RuntimeModelAvailabilityStore {
  return readStore();
}

export function getRuntimeModelAvailability(
  ref: string,
): RuntimeModelAvailabilityEntry | undefined {
  return readStore().models[ref];
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
}): RuntimeModelAvailabilityStatus {
  const described = describeFailoverError(params.error);
  const message = described.message.trim();
  let status: RuntimeModelAvailabilityStatus = "temporary_unavailable";

  if (described.reason === "auth") {
    status = "auth_invalid";
  } else if (described.reason === "rate_limit" || described.reason === "timeout") {
    status = "temporary_unavailable";
  } else if (isUnsupportedMessage(message) || described.status === 404) {
    status = "unsupported";
  }

  const store = readStore();
  store.models[params.ref] = {
    status,
    lastCheckedAt: Date.now(),
    ...(message ? { lastError: message } : {}),
  };
  writeStore(store);
  return status;
}

import { setTimeout as delay } from "node:timers/promises";

export type BackoffPolicy = {
  initialMs: number;
  maxMs: number;
  factor: number;
  jitter: number;
};

export const BACKOFF_PRESETS = {
  signalSseReconnect: {
    initialMs: 1_000,
    maxMs: 10_000,
    factor: 2,
    jitter: 0.2,
  },
  runnerReconnect: {
    initialMs: 2_000,
    maxMs: 30_000,
    factor: 1.8,
    jitter: 0.25,
  },
  channelRestart: {
    initialMs: 5_000,
    maxMs: 5 * 60_000,
    factor: 2,
    jitter: 0.1,
  },
} satisfies Record<string, BackoffPolicy>;

export type BackoffPresetName = keyof typeof BACKOFF_PRESETS;

export function resolveBackoffPolicy(
  preset: BackoffPresetName,
  overrides?: Partial<BackoffPolicy>,
): BackoffPolicy {
  return {
    ...BACKOFF_PRESETS[preset],
    ...overrides,
  };
}

export function computeBackoff(policy: BackoffPolicy, attempt: number) {
  const base = policy.initialMs * policy.factor ** Math.max(attempt - 1, 0);
  const jitter = base * policy.jitter * Math.random();
  return Math.min(policy.maxMs, Math.round(base + jitter));
}

export async function sleepWithAbort(ms: number, abortSignal?: AbortSignal) {
  if (ms <= 0) {
    return;
  }
  try {
    await delay(ms, undefined, { signal: abortSignal });
  } catch (err) {
    if (abortSignal?.aborted) {
      throw new Error("aborted", { cause: err });
    }
    throw err;
  }
}

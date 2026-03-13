import { describe, expect, it } from "vitest";
import { BACKOFF_PRESETS, resolveBackoffPolicy } from "./backoff.js";

describe("backoff presets", () => {
  it("exposes stable shared reconnect presets", () => {
    expect(BACKOFF_PRESETS.signalSseReconnect).toEqual({
      initialMs: 1_000,
      maxMs: 10_000,
      factor: 2,
      jitter: 0.2,
    });
    expect(BACKOFF_PRESETS.runnerReconnect).toEqual({
      initialMs: 2_000,
      maxMs: 30_000,
      factor: 1.8,
      jitter: 0.25,
    });
    expect(BACKOFF_PRESETS.channelRestart).toEqual({
      initialMs: 5_000,
      maxMs: 300_000,
      factor: 2,
      jitter: 0.1,
    });
  });

  it("merges overrides without mutating the preset", () => {
    const merged = resolveBackoffPolicy("runnerReconnect", {
      maxMs: 45_000,
    });
    expect(merged).toEqual({
      initialMs: 2_000,
      maxMs: 45_000,
      factor: 1.8,
      jitter: 0.25,
    });
    expect(BACKOFF_PRESETS.runnerReconnect.maxMs).toBe(30_000);
  });
});

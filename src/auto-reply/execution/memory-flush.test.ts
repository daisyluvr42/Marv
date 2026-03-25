import { describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";

vi.mock("../../agents/gateway.js", async () => {
  const { resolveCronStyleNow } = await vi.importActual<
    typeof import("../../agents/current-time.js")
  >("../../agents/current-time.js");
  const gateway = {
    runner: {
      runEmbedded: vi.fn(),
      runCli: vi.fn(),
      runWithFallback: vi.fn(),
      queueEmbeddedMessage: vi.fn(),
      abortEmbeddedRun: vi.fn(),
      isEmbeddedRunActive: vi.fn(),
      isEmbeddedRunStreaming: vi.fn(),
      resolveEmbeddedSessionLane: vi.fn(),
    },
    errors: {
      isContextOverflowError: vi.fn(),
      isLikelyContextOverflowError: vi.fn(),
      isCompactionFailureError: vi.fn(),
      isTransientHttpError: vi.fn(),
      sanitizeUserFacingText: vi.fn((t: string) => t),
    },
    models: {
      isCliProvider: vi.fn(),
      lookupContextTokens: vi.fn().mockReturnValue(undefined),
      loadModelCatalog: vi.fn(async () => []),
      hasConfiguredModelSelections: vi.fn(),
      resolveModelAuthMode: vi.fn(),
      resolveRuntimeModelPlan: vi.fn(),
      applyThinkingModelPreferences: vi.fn(),
      buildAllowedModelSet: vi.fn(),
      modelKey: vi.fn(),
      normalizeProviderId: vi.fn(),
      resolveModelRefFromString: vi.fn(),
      resolveThinkingDefault: vi.fn(),
    },
    auth: {
      clearSessionAuthProfileOverride: vi.fn(),
      resolveSessionAuthProfileOverride: vi.fn(),
      getCliSessionId: vi.fn(),
      ensureAuthProfileStore: vi.fn(),
    },
    sandbox: {
      resolveSandboxRuntimeStatus: vi.fn(),
      resolveSandboxConfigForAgent: vi.fn(),
    },
    constants: {
      DEFAULT_CONTEXT_TOKENS: 200_000,
      DEFAULT_PI_COMPACTION_RESERVE_TOKENS_FLOOR: 20_000,
    },
    utils: {
      resolveCronStyleNow,
      hasNonzeroUsage: vi.fn(),
    },
  };
  return { agents: gateway, getAgentGateway: () => gateway };
});

import { resolveMemoryFlushPromptForRun } from "./memory-flush.js";

describe("resolveMemoryFlushPromptForRun", () => {
  const cfg = {
    agents: {
      defaults: {
        userTimezone: "America/New_York",
        timeFormat: "12",
      },
    },
  } as MarvConfig;

  it("replaces YYYY-MM-DD using user timezone and appends current time", () => {
    const prompt = resolveMemoryFlushPromptForRun({
      prompt: "Store durable notes with memory_write on YYYY-MM-DD",
      cfg,
      nowMs: Date.UTC(2026, 1, 16, 15, 0, 0),
    });

    expect(prompt).toContain("2026-02-16");
    expect(prompt).toContain("memory_write");
    expect(prompt).toContain("Current time:");
    expect(prompt).toContain("(America/New_York)");
  });

  it("does not append a duplicate current time line", () => {
    const prompt = resolveMemoryFlushPromptForRun({
      prompt: "Store notes.\nCurrent time: already present",
      cfg,
      nowMs: Date.UTC(2026, 1, 16, 15, 0, 0),
    });

    expect(prompt).toContain("Current time: already present");
    expect((prompt.match(/Current time:/g) ?? []).length).toBe(1);
  });
});

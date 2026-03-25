import { vi } from "vitest";

vi.mock("../../agents/runner/pi-embedded.js", () => ({
  abortEmbeddedPiRun: vi.fn().mockReturnValue(false),
  runEmbeddedPiAgent: vi.fn(),
  queueEmbeddedPiMessage: vi.fn().mockReturnValue(false),
  resolveEmbeddedSessionLane: (key: string) => `session:${key.trim() || "main"}`,
  isEmbeddedPiRunActive: vi.fn().mockReturnValue(false),
  isEmbeddedPiRunStreaming: vi.fn().mockReturnValue(false),
}));

vi.mock("../../agents/model/model-catalog.js", () => ({
  loadModelCatalog: vi.fn(),
}));

vi.mock("../../agents/gateway.js", () => {
  const gateway = {
    runner: {
      runEmbedded: vi.fn(),
      runCli: vi.fn(),
      runWithFallback: vi.fn(
        async ({
          provider,
          model,
          run,
        }: {
          provider: string;
          model: string;
          run: (p: string, m: string) => Promise<unknown>;
        }) => ({
          result: await run(provider, model),
          provider,
          model,
        }),
      ),
      queueEmbeddedMessage: vi.fn().mockReturnValue(false),
      abortEmbeddedRun: vi.fn().mockReturnValue(false),
      isEmbeddedRunActive: vi.fn().mockReturnValue(false),
      isEmbeddedRunStreaming: vi.fn().mockReturnValue(false),
      resolveEmbeddedSessionLane: (key: string) => `session:${key.trim() || "main"}`,
    },
    errors: {
      isContextOverflowError: vi.fn().mockReturnValue(false),
      isLikelyContextOverflowError: vi.fn().mockReturnValue(false),
      isCompactionFailureError: vi.fn().mockReturnValue(false),
      isTransientHttpError: vi.fn().mockReturnValue(false),
      sanitizeUserFacingText: vi.fn((t: string) => t),
    },
    models: {
      isCliProvider: vi.fn().mockReturnValue(false),
      lookupContextTokens: vi.fn().mockReturnValue(200_000),
      loadModelCatalog: vi.fn(async () => []),
      hasConfiguredModelSelections: vi.fn().mockReturnValue(false),
      resolveModelAuthMode: vi.fn().mockReturnValue("api-key"),
      resolveRuntimeModelPlan: vi.fn().mockReturnValue({ candidates: [], pool: "default" }),
      applyThinkingModelPreferences: vi.fn((c: unknown) => c),
      buildAllowedModelSet: vi.fn().mockReturnValue(new Set()),
      modelKey: vi.fn((p: string, m: string) => `${p}/${m}`),
      normalizeProviderId: vi.fn((p: string) => p),
      resolveModelRefFromString: vi.fn(),
      resolveThinkingDefault: vi.fn(),
    },
    auth: {
      clearSessionAuthProfileOverride: vi.fn(),
      resolveSessionAuthProfileOverride: vi.fn().mockResolvedValue(undefined),
      getCliSessionId: vi.fn(),
      ensureAuthProfileStore: vi.fn(async () => ({})),
    },
    sandbox: {
      resolveSandboxRuntimeStatus: vi.fn().mockReturnValue({ sandboxed: false }),
      resolveSandboxConfigForAgent: vi.fn().mockReturnValue({}),
    },
    constants: {
      DEFAULT_CONTEXT_TOKENS: 200_000,
      DEFAULT_PI_COMPACTION_RESERVE_TOKENS_FLOOR: 20_000,
    },
    utils: {
      resolveCronStyleNow: vi.fn((_cfg: unknown, nowMs: number) => ({
        userTimezone: "UTC",
        formattedTime: new Date(nowMs).toISOString(),
        timeLine: `Current time: ${new Date(nowMs).toISOString()} (UTC)`,
      })),
      hasNonzeroUsage: vi.fn().mockReturnValue(false),
    },
  };
  return { agents: gateway, getAgentGateway: () => gateway };
});

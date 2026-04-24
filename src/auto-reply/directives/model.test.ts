import { describe, expect, it, vi } from "vitest";
import type { ModelAliasIndex } from "../../agents/model/model-resolve.js";
import type { MarvConfig } from "../../core/config/config.js";
import type { SessionEntry } from "../../core/config/sessions.js";
import { handleDirectiveOnly } from "./apply.js";
import { parseInlineDirectives } from "./index.js";
import { maybeHandleModelDirectiveInfo, resolveModelSelectionFromDirective } from "./model.js";

vi.mock("../../agents/gateway.js", async () => {
  const modelSel = await vi.importActual<typeof import("../../agents/model/model-resolve.js")>(
    "../../agents/model/model-resolve.js",
  );
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
      isCliProvider: vi.fn().mockReturnValue(false),
      lookupContextTokens: vi.fn().mockReturnValue(undefined),
      loadModelCatalog: vi.fn(async () => []),
      hasConfiguredModelSelections: vi.fn().mockReturnValue(false),
      resolveModelAuthMode: vi.fn().mockReturnValue("api-key"),
      resolveRuntimeModelPlan: vi.fn().mockReturnValue({ candidates: [], pool: "default" }),
      applyThinkingModelPreferences: vi.fn((c: unknown) => c),
      buildAllowedModelSet: vi.fn().mockReturnValue({ allowedKeys: new Set(), allowedCatalog: [] }),
      modelKey: modelSel.modelKey,
      normalizeProviderId: modelSel.normalizeProviderId,
      resolveModelRefFromString: modelSel.resolveModelRefFromString,
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
      resolveCronStyleNow: vi.fn(),
      hasNonzeroUsage: vi.fn(),
    },
  };
  return { agents: gateway, getAgentGateway: () => gateway };
});

// Mock dependencies for directive handling persistence.
vi.mock("../../agents/agent-scope.js", () => ({
  resolveAgentConfig: vi.fn(() => ({})),
  resolveAgentDir: vi.fn(() => "/tmp/agent"),
  resolveSessionAgentId: vi.fn(() => "main"),
}));

vi.mock("../../agents/sandbox/sandbox.js", () => ({
  resolveSandboxRuntimeStatus: vi.fn(() => ({ sandboxed: false })),
}));

vi.mock("../../core/config/sessions.js", () => ({
  updateSessionStore: vi.fn(async () => {}),
}));

vi.mock("../../infra/system-events.js", () => ({
  enqueueSystemEvent: vi.fn(),
}));

function baseAliasIndex(): ModelAliasIndex {
  return { byAlias: new Map(), byKey: new Map() };
}

function baseConfig(): MarvConfig {
  return {
    commands: { text: true },
    agents: { defaults: {} },
  } as unknown as MarvConfig;
}

describe("/model chat UX", () => {
  it("shows summary for /model with no args", async () => {
    const directives = parseInlineDirectives("/model");
    const cfg = { commands: { text: true } } as unknown as MarvConfig;

    const reply = await maybeHandleModelDirectiveInfo({
      directives,
      cfg,
      agentDir: "/tmp/agent",
      activeAgentId: "main",
      sessionEntry: undefined,
      provider: "anthropic",
      model: "claude-opus-4-5",
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-5",
      poolName: "default",
      runtimeCandidates: [],
      aliasIndex: baseAliasIndex(),
      allowedModelCatalog: [],
      resetModelOverride: false,
    });

    expect(reply?.text).toContain("Current:");
    expect(reply?.text).toContain("Browse: /models");
    expect(reply?.text).toContain("Switch: /model <provider/model>");
  });

  it("auto-applies closest match for typos", () => {
    const directives = parseInlineDirectives("/model anthropic/claud-opus-4-5");
    const cfg = { commands: { text: true } } as unknown as MarvConfig;

    const resolved = resolveModelSelectionFromDirective({
      directives,
      cfg,
      agentDir: "/tmp/agent",
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-5",
      aliasIndex: baseAliasIndex(),
      allowedModelKeys: new Set(["anthropic/claude-opus-4-5"]),
      allowedModelCatalog: [{ provider: "anthropic", id: "claude-opus-4-5" }],
      provider: "anthropic",
    });

    expect(resolved.modelSelection).toEqual({
      provider: "anthropic",
      model: "claude-opus-4-5",
      isDefault: true,
    });
    expect(resolved.errorText).toBeUndefined();
  });

  it("shows mode and pool details for /model status", async () => {
    const directives = parseInlineDirectives("/model status");
    const reply = await maybeHandleModelDirectiveInfo({
      directives,
      cfg: baseConfig(),
      agentDir: "/tmp/agent",
      activeAgentId: "main",
      sessionEntry: {
        sessionId: "s1",
        updatedAt: Date.now(),
        selectionMode: "manual",
        manualModelRef: "openai/gpt-4o",
      },
      provider: "openai",
      model: "gpt-4o",
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-5",
      poolName: "default",
      runtimeCandidates: [
        {
          ref: "openai/gpt-4o",
          provider: "openai",
          model: "gpt-4o",
          location: "cloud",
          tier: "standard",
          capabilities: ["text", "vision"],
          priority: 0,
          enabled: true,
          available: true,
          aliases: [],
        },
      ],
      aliasIndex: baseAliasIndex(),
      allowedModelCatalog: [],
      resetModelOverride: false,
    });

    expect(reply?.text).toContain("Mode: manual");
    expect(reply?.text).toContain("Pool: default");
    expect(reply?.text).toContain("Manual selection: openai/gpt-4o");
  });
});

describe("handleDirectiveOnly model persist behavior (fixes #1435)", () => {
  const allowedModelKeys = new Set(["anthropic/claude-opus-4-5", "openai/gpt-4o"]);
  const allowedModelCatalog = [
    { provider: "anthropic", id: "claude-opus-4-5", name: "Claude Opus 4.5" },
    { provider: "openai", id: "gpt-4o", name: "GPT-4o" },
  ];
  const sessionKey = "agent:main:dm:1";
  const storePath = "/tmp/sessions.json";

  type HandleParams = Parameters<typeof handleDirectiveOnly>[0];

  function createSessionEntry(overrides?: Partial<SessionEntry>): SessionEntry {
    return {
      sessionId: "s1",
      updatedAt: Date.now(),
      ...overrides,
    };
  }

  function createHandleParams(overrides: Partial<HandleParams>): HandleParams {
    const entryOverride = overrides.sessionEntry;
    const storeOverride = overrides.sessionStore;
    const entry = entryOverride ?? createSessionEntry();
    const store = storeOverride ?? ({ [sessionKey]: entry } as const);
    const { sessionEntry: _ignoredEntry, sessionStore: _ignoredStore, ...rest } = overrides;

    return {
      cfg: baseConfig(),
      directives: rest.directives ?? parseInlineDirectives(""),
      sessionKey,
      storePath,
      elevatedEnabled: false,
      elevatedAllowed: false,
      defaultProvider: "anthropic",
      defaultModel: "claude-opus-4-5",
      aliasIndex: baseAliasIndex(),
      allowedModelKeys,
      allowedModelCatalog,
      poolName: "default",
      candidates: [],
      resetModelOverride: false,
      provider: "anthropic",
      model: "claude-opus-4-5",
      initialModelLabel: "anthropic/claude-opus-4-5",
      formatModelSwitchEvent: (label) => `Switched to ${label}`,
      ...rest,
      sessionEntry: entry,
      sessionStore: store,
    };
  }

  it("shows success message when session state is available", async () => {
    const directives = parseInlineDirectives("/model openai/gpt-4o");
    const sessionEntry = createSessionEntry();
    const result = await handleDirectiveOnly(
      createHandleParams({
        directives,
        sessionEntry,
      }),
    );

    expect(result?.text).toContain("Model set to");
    expect(result?.text).toContain("openai/gpt-4o");
    expect(result?.text).not.toContain("failed");
  });

  it("shows no model message when no /model directive", async () => {
    const directives = parseInlineDirectives("hello world");
    const sessionEntry = createSessionEntry();
    const result = await handleDirectiveOnly(
      createHandleParams({
        directives,
        sessionEntry,
      }),
    );

    expect(result?.text ?? "").not.toContain("Model set to");
    expect(result?.text ?? "").not.toContain("failed");
  });

  it("persists thinkingLevel=off (does not clear)", async () => {
    const directives = parseInlineDirectives("/think off");
    const sessionEntry = createSessionEntry({ thinkingLevel: "low" });
    const sessionStore = { [sessionKey]: sessionEntry };
    const result = await handleDirectiveOnly(
      createHandleParams({
        directives,
        sessionEntry,
        sessionStore,
      }),
    );

    expect(result?.text ?? "").not.toContain("failed");
    expect(sessionEntry.thinkingLevel).toBe("off");
    expect(sessionStore["agent:main:dm:1"]?.thinkingLevel).toBe("off");
  });
});

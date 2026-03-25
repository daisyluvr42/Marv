import { beforeEach, describe, expect, it, vi } from "vitest";
import { runPreparedReply } from "./get-reply-run.js";

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
      resolveEmbeddedSessionLane: vi.fn().mockReturnValue("session:session-key"),
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
      lookupContextTokens: vi.fn().mockReturnValue(undefined),
      loadModelCatalog: vi.fn(async () => []),
      hasConfiguredModelSelections: vi.fn().mockReturnValue(false),
      resolveModelAuthMode: vi.fn().mockReturnValue("api-key"),
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

vi.mock("../../core/config/sessions.js", () => ({
  resolveGroupSessionKey: vi.fn().mockReturnValue(undefined),
  resolveSessionFilePath: vi.fn().mockReturnValue("/tmp/session.jsonl"),
  resolveSessionFilePathOptions: vi.fn().mockReturnValue({}),
  updateSessionStore: vi.fn(),
}));

vi.mock("../../globals.js", () => ({
  logVerbose: vi.fn(),
}));

vi.mock("../../process/command-queue.js", () => ({
  clearCommandLane: vi.fn().mockReturnValue(0),
  getQueueSize: vi.fn().mockReturnValue(0),
}));

vi.mock("../../routing/session-key.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../routing/session-key.js")>();
  return {
    ...actual,
    normalizeMainKey: vi.fn().mockReturnValue("main"),
  };
});

vi.mock("../../utils/provider-utils.js", () => ({
  isReasoningTagProvider: vi.fn().mockReturnValue(false),
}));

vi.mock("../command-detection.js", () => ({
  hasControlCommand: vi.fn().mockReturnValue(false),
}));

vi.mock("./runner.js", () => ({
  runReplyAgent: vi.fn().mockResolvedValue({ text: "ok" }),
}));

vi.mock("../delivery/body.js", () => ({
  applySessionHints: vi.fn().mockImplementation(async ({ baseBody }) => baseBody),
}));

vi.mock("../inbound/groups.js", () => ({
  buildGroupIntro: vi.fn().mockReturnValue(""),
  buildGroupChatContext: vi.fn().mockReturnValue(""),
  resolveGroupPersona: vi.fn().mockReturnValue(undefined),
}));

vi.mock("../inbound/meta.js", () => ({
  buildInboundMetaSystemPrompt: vi.fn().mockReturnValue(""),
  buildInboundUserContextPrefix: vi.fn().mockReturnValue(""),
}));

vi.mock("../queue/index.js", () => ({
  resolveQueueSettings: vi.fn().mockReturnValue({ mode: "followup" }),
}));

vi.mock("../delivery/route.js", () => ({
  routeReply: vi.fn(),
}));

vi.mock("../session/updates.js", () => ({
  ensureSkillSnapshot: vi.fn().mockImplementation(async ({ sessionEntry, systemSent }) => ({
    sessionEntry,
    systemSent,
    skillsSnapshot: undefined,
  })),
  prependSystemEvents: vi.fn().mockImplementation(async ({ prefixedBodyBase }) => prefixedBodyBase),
}));

vi.mock("../delivery/typing-mode.js", () => ({
  resolveTypingMode: vi.fn().mockReturnValue("off"),
}));

import { runReplyAgent } from "./runner.js";

function baseParams(
  overrides: Partial<Parameters<typeof runPreparedReply>[0]> = {},
): Parameters<typeof runPreparedReply>[0] {
  return {
    ctx: {
      Body: "",
      RawBody: "",
      CommandBody: "",
      ThreadHistoryBody: "Earlier message in this thread",
      OriginatingChannel: "slack",
      OriginatingTo: "C123",
      ChatType: "group",
    },
    sessionCtx: {
      Body: "",
      BodyStripped: "",
      ThreadHistoryBody: "Earlier message in this thread",
      MediaPath: "/tmp/input.png",
      Provider: "slack",
      ChatType: "group",
      OriginatingChannel: "slack",
      OriginatingTo: "C123",
    },
    cfg: { session: {}, channels: {}, agents: { defaults: {} } },
    agentId: "default",
    agentDir: "/tmp/agent",
    agentCfg: {},
    sessionCfg: {},
    commandAuthorized: true,
    command: {
      isAuthorizedSender: true,
      abortKey: "session-key",
      ownerList: [],
      senderIsOwner: false,
    } as never,
    commandSource: "",
    allowTextCommands: true,
    directives: {
      hasThinkDirective: false,
      thinkLevel: undefined,
    } as never,
    defaultActivation: "always",
    resolvedThinkLevel: "high",
    resolvedVerboseLevel: "off",
    resolvedReasoningLevel: "off",
    resolvedElevatedLevel: "off",
    elevatedEnabled: false,
    elevatedAllowed: false,
    blockStreamingEnabled: false,
    resolvedBlockStreamingBreak: "message_end",
    modelState: {
      resolveDefaultThinkingLevel: async () => "medium",
      candidates: [],
    } as never,
    provider: "anthropic",
    model: "claude-opus-4-1",
    typing: {
      onReplyStart: vi.fn().mockResolvedValue(undefined),
      cleanup: vi.fn(),
    } as never,
    defaultProvider: "anthropic",
    defaultModel: "claude-opus-4-1",
    timeoutMs: 30_000,
    isNewSession: true,
    resetTriggered: false,
    systemSent: true,
    sessionKey: "session-key",
    workspaceDir: "/tmp/workspace",
    abortedLastRun: false,
    ...overrides,
  };
}

describe("runPreparedReply media-only handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("allows media-only prompts and preserves thread context in queued followups", async () => {
    const result = await runPreparedReply(baseParams());
    expect(result).toEqual({ text: "ok" });

    const call = vi.mocked(runReplyAgent).mock.calls[0]?.[0];
    expect(call).toBeTruthy();
    expect(call?.followupRun.prompt).toContain("[Thread history - for context]");
    expect(call?.followupRun.prompt).toContain("Earlier message in this thread");
    expect(call?.followupRun.prompt).toContain("[User sent media without caption]");
  });

  it("injects privacy guard directive into extra system prompt when context requires protection", async () => {
    await runPreparedReply(baseParams());

    const call = vi.mocked(runReplyAgent).mock.calls[0]?.[0];
    expect(call?.followupRun.run.extraSystemPrompt).toContain("PRIVACY GUARD ACTIVE");
  });

  it("redacts sensitive output before returning replies", async () => {
    vi.mocked(runReplyAgent).mockResolvedValueOnce({
      text: "my secret key is sk-abc123def456ghi789jkl012mno345",
    });

    const result = await runPreparedReply(baseParams());

    expect(result).toEqual({
      text: "my secret key is [REDACTED:api_keys]",
    });
  });

  it("skips output redaction when privacy output scanning is disabled", async () => {
    vi.mocked(runReplyAgent).mockResolvedValueOnce({
      text: "my secret key is sk-abc123def456ghi789jkl012mno345",
    });

    const result = await runPreparedReply(
      baseParams({
        cfg: {
          session: {},
          channels: {},
          agents: { defaults: {} },
          autonomy: {
            privacy: {
              outputScan: false,
            },
          },
        },
      }),
    );

    expect(result).toEqual({
      text: "my secret key is sk-abc123def456ghi789jkl012mno345",
    });
  });

  it("returns the empty-body reply when there is no text and no media", async () => {
    const result = await runPreparedReply(
      baseParams({
        ctx: {
          Body: "",
          RawBody: "",
          CommandBody: "",
        },
        sessionCtx: {
          Body: "",
          BodyStripped: "",
          Provider: "slack",
        },
      }),
    );

    expect(result).toEqual({
      text: "I didn't receive any text in your message. Please resend or add a caption.",
    });
    expect(vi.mocked(runReplyAgent)).not.toHaveBeenCalled();
  });
});

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../core/config/config.js";
import { createTempHomeHarness, makeReplyConfig } from "./reply.test-harness.js";

const runEmbeddedPiAgentMock = vi.fn();

vi.mock("../agents/model/model-fallback.js", () => ({
  runWithModelFallback: async ({
    provider,
    model,
    run,
  }: {
    provider: string;
    model: string;
    run: (provider: string, model: string) => Promise<unknown>;
  }) => ({
    result: await run(provider, model),
    provider,
    model,
  }),
}));

vi.mock("../agents/runner/pi-embedded.js", () => ({
  abortEmbeddedPiRun: vi.fn().mockReturnValue(false),
  runEmbeddedPiAgent: (params: unknown) => runEmbeddedPiAgentMock(params),
  queueEmbeddedPiMessage: vi.fn().mockReturnValue(false),
  resolveEmbeddedSessionLane: (key: string) => `session:${key.trim() || "main"}`,
  isEmbeddedPiRunActive: vi.fn().mockReturnValue(false),
  isEmbeddedPiRunStreaming: vi.fn().mockReturnValue(false),
}));

const webMocks = vi.hoisted(() => ({
  webAuthExists: vi.fn().mockResolvedValue(true),
  getWebAuthAgeMs: vi.fn().mockReturnValue(120_000),
  readWebSelfId: vi.fn().mockReturnValue({ e164: "+1999" }),
}));

vi.mock("../channels/web/session.js", () => webMocks);

import { getReplyFromConfig } from "./reply.js";

const { withTempHome } = createTempHomeHarness({
  prefix: "marv-typing-",
  beforeEachCase: () => runEmbeddedPiAgentMock.mockClear(),
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("getReplyFromConfig typing (heartbeat)", () => {
  beforeEach(() => {
    vi.stubEnv("MARV_TEST_FAST", "1");
  });

  it("starts typing for normal runs", async () => {
    await withTempHome(async (home) => {
      runEmbeddedPiAgentMock.mockResolvedValueOnce({
        payloads: [{ text: "ok" }],
        meta: {},
      });
      const onReplyStart = vi.fn();

      await getReplyFromConfig(
        { Body: "hi", From: "+1000", To: "+2000", Provider: "whatsapp" },
        { onReplyStart, isHeartbeat: false },
        makeReplyConfig(home) as unknown as MarvConfig,
      );

      expect(onReplyStart).toHaveBeenCalled();
    });
  });

  it("does not start typing for heartbeat runs", async () => {
    await withTempHome(async (home) => {
      runEmbeddedPiAgentMock.mockResolvedValueOnce({
        payloads: [{ text: "ok" }],
        meta: {},
      });
      const onReplyStart = vi.fn();

      await getReplyFromConfig(
        { Body: "hi", From: "+1000", To: "+2000", Provider: "whatsapp" },
        { onReplyStart, isHeartbeat: true },
        makeReplyConfig(home) as unknown as MarvConfig,
      );

      expect(onReplyStart).not.toHaveBeenCalled();
    });
  });

  it("does not start typing when heartbeat semantics come from runMode", async () => {
    await withTempHome(async (home) => {
      runEmbeddedPiAgentMock.mockResolvedValueOnce({
        payloads: [{ text: "ok" }],
        meta: {},
      });
      const onReplyStart = vi.fn();

      await getReplyFromConfig(
        { Body: "hi", From: "+1000", To: "+2000", Provider: "whatsapp" },
        {
          onReplyStart,
          runMode: {
            kind: "heartbeat",
            reason: "cron",
            ackToken: "HEARTBEAT_OK",
            maxAckChars: 300,
            visibility: "hidden",
          },
        },
        makeReplyConfig(home) as unknown as MarvConfig,
      );

      expect(onReplyStart).not.toHaveBeenCalled();
    });
  });
});

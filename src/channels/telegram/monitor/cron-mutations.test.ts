import type { Bot } from "grammy";
import { describe, expect, it, vi } from "vitest";
import { TelegramCronMutationHandler } from "./cron-mutations.js";

vi.mock("../../../core/gateway/client.js", () => ({
  GatewayClient: class {
    start() {}
    stop() {}
  },
}));

vi.mock("../../../logger.js", () => ({
  logDebug: vi.fn(),
  logError: vi.fn(),
}));

function createBotMock() {
  const sendMessageMock = vi.fn().mockResolvedValue(undefined);
  return {
    bot: {
      api: {
        sendMessage: sendMessageMock,
      },
    } as unknown as Bot,
    sendMessageMock,
  };
}

type TelegramCronMutationInternals = {
  handleCronMutation: (event: Record<string, unknown>) => Promise<void>;
};

describe("TelegramCronMutationHandler", () => {
  it("sends readable cron mutation notifications to configured approvers", async () => {
    const { bot, sendMessageMock } = createBotMock();
    const handler = new TelegramCronMutationHandler({
      bot,
      config: {
        approvers: ["123456"],
        agentFilter: ["main"],
      },
      cfg: {},
    });

    await (handler as unknown as TelegramCronMutationInternals).handleCronMutation({
      action: "updated",
      jobId: "job-1",
      jobName: "Morning brief",
      agentId: "main",
      sessionTarget: "isolated",
      deliveryMode: "announce",
      nextRunAtMs: Date.parse("2026-04-01T00:00:00.000Z"),
    });

    expect(sendMessageMock).toHaveBeenCalledTimes(1);
    const [, text] = sendMessageMock.mock.calls[0] as [number, string];
    expect(text).toContain("Cron job updated");
    expect(text).toContain("Morning brief");
    expect(text).toContain("job-1");
  });

  it("requires configured approvers", () => {
    const { bot } = createBotMock();
    const handler = new TelegramCronMutationHandler({
      bot,
      config: undefined,
      cfg: {},
    });

    expect(
      handler.shouldHandle({
        action: "added",
        jobId: "job-1",
      }),
    ).toBe(false);
  });
});

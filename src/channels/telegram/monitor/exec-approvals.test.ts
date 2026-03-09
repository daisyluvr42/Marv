import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { Bot } from "grammy";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearSessionStoreCacheForTest } from "../../../core/config/sessions.js";
import type { TelegramExecApprovalConfig } from "../../../core/config/types.telegram.js";
import { TelegramExecApprovalHandler, type ExecApprovalRequest } from "./exec-approvals.js";

const STORE_PATH = path.join(
  os.tmpdir(),
  `marv-telegram-exec-approvals-test-${process.pid}-${Math.random().toString(36).slice(2)}.json`,
);

const writeStore = (store: Record<string, unknown>) => {
  fs.writeFileSync(STORE_PATH, `${JSON.stringify(store, null, 2)}\n`, "utf8");
  clearSessionStoreCacheForTest();
};

beforeEach(() => {
  writeStore({});
});

afterEach(() => {
  vi.useRealTimers();
});

vi.mock("../../../core/gateway/client.js", () => ({
  GatewayClient: class {
    start() {}
    stop() {}
    async request() {
      return { ok: true };
    }
  },
}));

vi.mock("../../../logger.js", () => ({
  logDebug: vi.fn(),
  logError: vi.fn(),
}));

function createBotMock() {
  return {
    api: {
      sendMessage: vi.fn().mockResolvedValue({
        chat: { id: 123456 },
        message_id: 77,
      }),
      editMessageText: vi.fn().mockResolvedValue(undefined),
    },
  } as unknown as Bot;
}

function createHandler(config: TelegramExecApprovalConfig, accountId = "default") {
  const bot = createBotMock();
  const handler = new TelegramExecApprovalHandler({
    bot,
    accountId,
    config,
    cfg: { session: { store: STORE_PATH } },
  });
  return { bot, handler };
}

function createRequest(
  overrides: Partial<ExecApprovalRequest["request"]> = {},
): ExecApprovalRequest {
  return {
    id: "test-id",
    request: {
      command: "echo hello",
      cwd: "/Users/daisyluvr/Documents/Marv",
      host: "gateway",
      agentId: "test-agent",
      sessionKey: "agent:test-agent:telegram:direct:123456",
      ...overrides,
    },
    createdAtMs: Date.now(),
    expiresAtMs: Date.now() + 60000,
  };
}

type TelegramExecApprovalInternals = {
  pending: Map<string, { timeoutId: NodeJS.Timeout }>;
  handleApprovalRequested: (request: ExecApprovalRequest) => Promise<void>;
};

function getInternals(handler: TelegramExecApprovalHandler): TelegramExecApprovalInternals {
  return handler as unknown as TelegramExecApprovalInternals;
}

function clearPendingTimeouts(handler: TelegramExecApprovalHandler) {
  const internals = getInternals(handler);
  for (const pending of internals.pending.values()) {
    clearTimeout(pending.timeoutId);
  }
  internals.pending.clear();
}

describe("TelegramExecApprovalHandler.shouldHandle", () => {
  it("returns false when disabled", () => {
    const { handler } = createHandler({ enabled: false, approvers: ["123456"] });
    expect(handler.shouldHandle(createRequest())).toBe(false);
  });

  it("filters by telegram account from the session store", () => {
    writeStore({
      "agent:test-agent:telegram:direct:123456": {
        sessionId: "sess",
        updatedAt: Date.now(),
        origin: { provider: "telegram", accountId: "secondary" },
        lastAccountId: "secondary",
      },
    });

    const { handler } = createHandler({ enabled: true, approvers: ["123456"] }, "default");
    expect(handler.shouldHandle(createRequest())).toBe(false);

    const matching = createHandler({ enabled: true, approvers: ["123456"] }, "secondary");
    expect(matching.handler.shouldHandle(createRequest())).toBe(true);
  });

  it("applies safe session filters", () => {
    const { handler } = createHandler({
      enabled: true,
      approvers: ["123456"],
      sessionFilter: ["^agent:.*:telegram:"],
    });

    expect(handler.shouldHandle(createRequest())).toBe(true);
    expect(
      handler.shouldHandle(
        createRequest({
          sessionKey: "agent:test-agent:slack:direct:123456",
        }),
      ),
    ).toBe(false);
  });
});

describe("TelegramExecApprovalHandler messages", () => {
  it("renders permission escalation details in the Telegram approval message", async () => {
    vi.useFakeTimers();
    const { bot, handler } = createHandler({ enabled: true, approvers: ["123456"] });
    const sendMessageMock = vi.spyOn(bot.api, "sendMessage");

    await getInternals(handler).handleApprovalRequested(
      createRequest({
        command: "request_escalation execute",
        kind: "permission-escalation",
        taskId: "task-42",
      }),
    );

    expect(sendMessageMock).toHaveBeenCalledTimes(1);
    const [, messageText] = sendMessageMock.mock.calls[0] as [number, string];
    expect(messageText).toContain("Permission Escalation Required");
    expect(messageText).toContain("Kind: permission-escalation");
    expect(messageText).toContain("Task: task-42");

    clearPendingTimeouts(handler);
  });
});

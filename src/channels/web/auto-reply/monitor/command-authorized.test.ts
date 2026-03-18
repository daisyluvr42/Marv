import { beforeEach, describe, expect, it, vi } from "vitest";

const readChannelAllowFromStore = vi.hoisted(() => vi.fn(async (): Promise<string[]> => []));

vi.mock("../../../../pairing/pairing-store.js", () => ({
  readChannelAllowFromStore,
}));

import { resolveWhatsAppCommandAuthorized } from "./command-authorized.js";

function createConfig(overrides?: Record<string, unknown>) {
  return {
    channels: {
      whatsapp: {},
    },
    ...overrides,
  } as never;
}

function createMessage(overrides?: Record<string, unknown>) {
  return {
    chatType: "group",
    senderE164: "+15550001111",
    from: "123@g.us",
    selfE164: "+15559990000",
    accountId: "default",
    ...overrides,
  } as never;
}

describe("resolveWhatsAppCommandAuthorized", () => {
  beforeEach(() => {
    readChannelAllowFromStore.mockReset();
    readChannelAllowFromStore.mockResolvedValue([]);
  });

  it("bypasses sender allowlists when access groups are disabled", async () => {
    await expect(
      resolveWhatsAppCommandAuthorized({
        cfg: createConfig({
          commands: { useAccessGroups: false },
          channels: { whatsapp: { allowFrom: ["+1999"] } },
        }),
        msg: createMessage({
          chatType: "group",
          senderE164: "+15550001111",
        }),
      }),
    ).resolves.toBe(true);
  });

  it("allows group control commands for wildcard group allowlists", async () => {
    await expect(
      resolveWhatsAppCommandAuthorized({
        cfg: createConfig({
          channels: { whatsapp: { groupAllowFrom: ["*"] } },
        }),
        msg: createMessage({
          chatType: "group",
          senderE164: "+15550001111",
        }),
      }),
    ).resolves.toBe(true);
  });

  it("blocks group control commands when the sender is outside the configured allowlist", async () => {
    await expect(
      resolveWhatsAppCommandAuthorized({
        cfg: createConfig({
          channels: { whatsapp: { groupAllowFrom: ["+15550002222"] } },
        }),
        msg: createMessage({
          chatType: "group",
          senderE164: "+15550001111",
        }),
      }),
    ).resolves.toBe(false);
  });

  it("falls back to the self phone for direct chats without explicit allowFrom", async () => {
    await expect(
      resolveWhatsAppCommandAuthorized({
        cfg: createConfig(),
        msg: createMessage({
          chatType: "direct",
          from: "+15559990000",
          senderE164: "+15559990000",
          selfE164: "+15559990000",
        }),
      }),
    ).resolves.toBe(true);
  });

  it("includes stored allowFrom values for direct chats", async () => {
    readChannelAllowFromStore.mockResolvedValueOnce(["+15550003333"]);

    await expect(
      resolveWhatsAppCommandAuthorized({
        cfg: createConfig(),
        msg: createMessage({
          chatType: "direct",
          from: "+15550003333",
          senderE164: "+15550003333",
        }),
      }),
    ).resolves.toBe(true);
  });
});

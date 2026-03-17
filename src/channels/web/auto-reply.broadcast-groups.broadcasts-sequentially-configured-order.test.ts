import "./test-helpers.js";
import { describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";
import {
  monitorWebChannelWithCapture,
  sendWebDirectInboundAndCollectSessionKeys,
} from "./auto-reply.broadcast-groups.test-harness.js";
import {
  installWebAutoReplyTestHomeHooks,
  installWebAutoReplyUnitTestHooks,
  resetLoadConfigMock,
  sendWebGroupInboundMessage,
  setLoadConfigMock,
} from "./auto-reply.test-harness.js";

installWebAutoReplyTestHomeHooks();

describe("broadcast groups", () => {
  installWebAutoReplyUnitTestHooks();

  it("broadcasts to main agent with single-agent config", async () => {
    setLoadConfigMock({
      channels: { whatsapp: { allowFrom: ["*"] } },
      agents: {
        defaults: { maxConcurrent: 10 },
      },
      broadcast: {
        strategy: "sequential",
        "+1000": ["main"],
      },
    } satisfies MarvConfig);

    const { seen, resolver } = await sendWebDirectInboundAndCollectSessionKeys();

    expect(resolver).toHaveBeenCalledTimes(1);
    expect(seen[0]).toContain("agent:main:");
    resetLoadConfigMock();
  });
  it("shares group history with main agent and clears after replying", async () => {
    setLoadConfigMock({
      channels: { whatsapp: { allowFrom: ["*"] } },
      agents: {
        defaults: { maxConcurrent: 10 },
      },
      broadcast: {
        strategy: "sequential",
        "123@g.us": ["main"],
      },
    } satisfies MarvConfig);

    const resolver = vi.fn().mockResolvedValue({ text: "ok" });

    const { spies, onMessage } = await monitorWebChannelWithCapture(resolver);

    await sendWebGroupInboundMessage({
      onMessage,
      spies,
      body: "hello group",
      id: "g1",
      senderE164: "+111",
      senderName: "Alice",
      selfE164: "+999",
    });

    expect(resolver).not.toHaveBeenCalled();

    await sendWebGroupInboundMessage({
      onMessage,
      spies,
      body: "@bot ping",
      id: "g2",
      senderE164: "+222",
      senderName: "Bob",
      mentionedJids: ["999@s.whatsapp.net"],
      selfE164: "+999",
      selfJid: "999@s.whatsapp.net",
    });

    expect(resolver).toHaveBeenCalledTimes(1);
    const payload = resolver.mock.calls[0][0] as {
      Body: string;
      SenderName?: string;
      SenderE164?: string;
      SenderId?: string;
    };
    expect(payload.Body).toContain("Chat messages since your last reply");
    expect(payload.Body).toContain("Alice (+111): hello group");
    // Message id hints are not included in prompts anymore.
    expect(payload.Body).not.toContain("[message_id:");
    expect(payload.Body).toContain("@bot ping");
    expect(payload.SenderName).toBe("Bob");
    expect(payload.SenderE164).toBe("+222");
    expect(payload.SenderId).toBe("+222");

    await sendWebGroupInboundMessage({
      onMessage,
      spies,
      body: "@bot ping 2",
      id: "g3",
      senderE164: "+333",
      senderName: "Clara",
      mentionedJids: ["999@s.whatsapp.net"],
      selfE164: "+999",
      selfJid: "999@s.whatsapp.net",
    });

    expect(resolver).toHaveBeenCalledTimes(2);
    const followup = resolver.mock.calls[1][0] as { Body: string };
    expect(followup.Body).not.toContain("Alice (+111): hello group");
    expect(followup.Body).not.toContain("Chat messages since your last reply");

    resetLoadConfigMock();
  });
});

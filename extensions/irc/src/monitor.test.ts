import { describe, expect, it } from "vitest";
import { resolveIrcInboundTarget } from "./monitor.js";

describe("irc monitor inbound target", () => {
  it("keeps channel target for group messages", () => {
    expect(
      resolveIrcInboundTarget({
        target: "#marv",
        senderNick: "alice",
      }),
    ).toEqual({
      isGroup: true,
      target: "#marv",
      rawTarget: "#marv",
    });
  });

  it("maps DM target to sender nick and preserves raw target", () => {
    expect(
      resolveIrcInboundTarget({
        target: "marv-bot",
        senderNick: "alice",
      }),
    ).toEqual({
      isGroup: false,
      target: "alice",
      rawTarget: "marv-bot",
    });
  });

  it("falls back to raw target when sender nick is empty", () => {
    expect(
      resolveIrcInboundTarget({
        target: "marv-bot",
        senderNick: " ",
      }),
    ).toEqual({
      isGroup: false,
      target: "marv-bot",
      rawTarget: "marv-bot",
    });
  });
});

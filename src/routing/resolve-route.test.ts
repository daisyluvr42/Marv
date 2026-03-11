import { describe, expect, test } from "vitest";
import type { MarvConfig } from "../core/config/config.js";
import { resolveAgentRoute } from "./resolve-route.js";

describe("resolveAgentRoute", () => {
  test("always resolves the durable main agent in single-agent mode", () => {
    const cfg: MarvConfig = {};
    const route = resolveAgentRoute({
      cfg,
      channel: "whatsapp",
      accountId: null,
      peer: { kind: "direct", id: "+15551234567" },
    });

    expect(route.agentId).toBe("main");
    expect(route.accountId).toBe("default");
    expect(route.matchedBy).toBe("default");
  });

  test("dmScope=main collapses direct sessions to the main key", () => {
    const route = resolveAgentRoute({
      cfg: {},
      channel: "whatsapp",
      peer: { kind: "direct", id: "+15551234567" },
    });

    expect(route.sessionKey).toBe("agent:main:main");
    expect(route.mainSessionKey).toBe("agent:main:main");
  });

  test("dmScope=per-peer isolates direct sessions by sender id", () => {
    const cfg: MarvConfig = {
      session: { dmScope: "per-peer" },
    };
    const route = resolveAgentRoute({
      cfg,
      channel: "whatsapp",
      peer: { kind: "direct", id: "+15551234567" },
    });

    expect(route.sessionKey).toBe("agent:main:direct:+15551234567");
  });

  test("dmScope=per-channel-peer keeps direct sessions channel-scoped", () => {
    const cfg: MarvConfig = {
      session: { dmScope: "per-channel-peer" },
    };
    const route = resolveAgentRoute({
      cfg,
      channel: "discord",
      peer: { kind: "direct", id: "alice" },
    });

    expect(route.sessionKey).toBe("agent:main:discord:direct:alice");
  });

  test("identityLinks still collapse linked direct identities", () => {
    const cfg: MarvConfig = {
      session: {
        dmScope: "per-peer",
        identityLinks: {
          alice: ["telegram:111111111", "discord:222222222222222222"],
        },
      },
    };
    const route = resolveAgentRoute({
      cfg,
      channel: "telegram",
      peer: { kind: "direct", id: "111111111" },
    });

    expect(route.sessionKey).toBe("agent:main:direct:alice");
  });

  test("channel and group peers still produce stable scoped keys", () => {
    const route = resolveAgentRoute({
      cfg: {},
      channel: "discord",
      accountId: "default",
      peer: { kind: "channel", id: "c1" },
    });

    expect(route.sessionKey).toBe("agent:main:discord:channel:c1");
  });
});

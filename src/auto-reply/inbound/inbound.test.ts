import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";
import type { GroupKeyResolution } from "../../core/config/sessions.js";
import { initSessionState } from "../session/init.js";
import {
  buildMentionRegexes,
  matchesMentionPatterns,
  normalizeMentionText,
} from "../support/mentions.js";
import {
  applyTemplate,
  type TurnContext,
  type SessionTemplateContext,
} from "../support/templating.js";
import { finalizeInboundContext } from "./context.js";
import { createInboundDebouncer } from "./debounce.js";
import { buildInboundDedupeKey, resetInboundDedupe, shouldSkipDuplicateInbound } from "./dedupe.js";
import { resolveGroupRequireMention } from "./groups.js";
import { normalizeInboundTextNewlines } from "./text.js";

describe("applyTemplate", () => {
  it("renders primitive values", () => {
    const ctx = { MessageSid: "sid", IsNewSession: "no" } as SessionTemplateContext;
    const overrides = ctx as Record<string, unknown>;
    overrides.MessageSid = 42;
    overrides.IsNewSession = true;

    expect(applyTemplate("sid={{MessageSid}} new={{IsNewSession}}", ctx)).toBe("sid=42 new=true");
  });

  it("renders arrays of primitives", () => {
    const ctx = { MediaPaths: ["a"] } as SessionTemplateContext;
    (ctx as Record<string, unknown>).MediaPaths = ["a", 2, true, null, { ok: false }];

    expect(applyTemplate("paths={{MediaPaths}}", ctx)).toBe("paths=a,2,true");
  });

  it("drops object values", () => {
    const ctx: SessionTemplateContext = { CommandArgs: { raw: "go" } };

    expect(applyTemplate("args={{CommandArgs}}", ctx)).toBe("args=");
  });

  it("renders missing placeholders as empty", () => {
    const ctx: SessionTemplateContext = {};

    expect(applyTemplate("missing={{Missing}}", ctx)).toBe("missing=");
  });
});

describe("normalizeInboundTextNewlines", () => {
  it("keeps real newlines", () => {
    expect(normalizeInboundTextNewlines("a\nb")).toBe("a\nb");
  });

  it("normalizes CRLF/CR to LF", () => {
    expect(normalizeInboundTextNewlines("a\r\nb")).toBe("a\nb");
    expect(normalizeInboundTextNewlines("a\rb")).toBe("a\nb");
  });

  it("preserves literal backslash-n sequences (Windows paths)", () => {
    // Windows paths like C:\Work\nxxx should NOT have \n converted to newlines
    expect(normalizeInboundTextNewlines("a\\nb")).toBe("a\\nb");
    expect(normalizeInboundTextNewlines("C:\\Work\\nxxx")).toBe("C:\\Work\\nxxx");
  });
});

describe("finalizeInboundContext", () => {
  it("fills BodyForAgent/BodyForCommands and normalizes newlines", () => {
    const ctx: TurnContext = {
      // Use actual CRLF for newline normalization test, not literal \n sequences
      Body: "a\r\nb\r\nc",
      RawBody: "raw\r\nline",
      ChatType: "channel",
      From: "whatsapp:group:123@g.us",
      GroupSubject: "Test",
    };

    const out = finalizeInboundContext(ctx);
    expect(out.Body).toBe("a\nb\nc");
    expect(out.RawBody).toBe("raw\nline");
    // Prefer clean text over legacy envelope-shaped Body when RawBody is present.
    expect(out.BodyForAgent).toBe("raw\nline");
    expect(out.BodyForCommands).toBe("raw\nline");
    expect(out.CommandAuthorized).toBe(false);
    expect(out.ChatType).toBe("channel");
    expect(out.ConversationLabel).toContain("Test");
  });

  it("preserves literal backslash-n in Windows paths", () => {
    const ctx: TurnContext = {
      Body: "C:\\Work\\nxxx\\README.md",
      RawBody: "C:\\Work\\nxxx\\README.md",
      ChatType: "direct",
      From: "web:user",
    };

    const out = finalizeInboundContext(ctx);
    expect(out.Body).toBe("C:\\Work\\nxxx\\README.md");
    expect(out.BodyForAgent).toBe("C:\\Work\\nxxx\\README.md");
    expect(out.BodyForCommands).toBe("C:\\Work\\nxxx\\README.md");
  });

  it("can force BodyForCommands to follow updated CommandBody", () => {
    const ctx: TurnContext = {
      Body: "base",
      BodyForCommands: "<media:audio>",
      CommandBody: "say hi",
      From: "signal:+15550001111",
      ChatType: "direct",
    };

    finalizeInboundContext(ctx, { forceBodyForCommands: true });
    expect(ctx.BodyForCommands).toBe("say hi");
  });

  it("fills MediaType/MediaTypes defaults only when media exists", () => {
    const withMedia: TurnContext = {
      Body: "hi",
      MediaPath: "/tmp/file.bin",
    };
    const outWithMedia = finalizeInboundContext(withMedia);
    expect(outWithMedia.MediaType).toBe("application/octet-stream");
    expect(outWithMedia.MediaTypes).toEqual(["application/octet-stream"]);

    const withoutMedia: TurnContext = { Body: "hi" };
    const outWithoutMedia = finalizeInboundContext(withoutMedia);
    expect(outWithoutMedia.MediaType).toBeUndefined();
    expect(outWithoutMedia.MediaTypes).toBeUndefined();
  });

  it("pads MediaTypes to match MediaPaths/MediaUrls length", () => {
    const ctx: TurnContext = {
      Body: "hi",
      MediaPaths: ["/tmp/a", "/tmp/b"],
      MediaTypes: ["image/png"],
    };
    const out = finalizeInboundContext(ctx);
    expect(out.MediaType).toBe("image/png");
    expect(out.MediaTypes).toEqual(["image/png", "application/octet-stream"]);
  });

  it("derives MediaType from MediaTypes when missing", () => {
    const ctx: TurnContext = {
      Body: "hi",
      MediaPath: "/tmp/a",
      MediaTypes: ["image/jpeg"],
    };
    const out = finalizeInboundContext(ctx);
    expect(out.MediaType).toBe("image/jpeg");
    expect(out.MediaTypes).toEqual(["image/jpeg"]);
  });
});

describe("inbound dedupe", () => {
  it("builds a stable key when MessageSid is present", () => {
    const ctx: TurnContext = {
      Provider: "telegram",
      OriginatingChannel: "telegram",
      OriginatingTo: "telegram:123",
      MessageSid: "42",
    };
    expect(buildInboundDedupeKey(ctx)).toBe("telegram|telegram:123|42");
  });

  it("skips duplicates with the same key", () => {
    resetInboundDedupe();
    const ctx: TurnContext = {
      Provider: "whatsapp",
      OriginatingChannel: "whatsapp",
      OriginatingTo: "whatsapp:+1555",
      MessageSid: "msg-1",
    };
    expect(shouldSkipDuplicateInbound(ctx, { now: 100 })).toBe(false);
    expect(shouldSkipDuplicateInbound(ctx, { now: 200 })).toBe(true);
  });

  it("does not dedupe when the peer changes", () => {
    resetInboundDedupe();
    const base: TurnContext = {
      Provider: "whatsapp",
      OriginatingChannel: "whatsapp",
      MessageSid: "msg-1",
    };
    expect(
      shouldSkipDuplicateInbound({ ...base, OriginatingTo: "whatsapp:+1000" }, { now: 100 }),
    ).toBe(false);
    expect(
      shouldSkipDuplicateInbound({ ...base, OriginatingTo: "whatsapp:+2000" }, { now: 200 }),
    ).toBe(false);
  });

  it("does not dedupe across session keys", () => {
    resetInboundDedupe();
    const base: TurnContext = {
      Provider: "whatsapp",
      OriginatingChannel: "whatsapp",
      OriginatingTo: "whatsapp:+1555",
      MessageSid: "msg-1",
    };
    expect(
      shouldSkipDuplicateInbound({ ...base, SessionKey: "agent:alpha:main" }, { now: 100 }),
    ).toBe(false);
    expect(
      shouldSkipDuplicateInbound({ ...base, SessionKey: "agent:bravo:main" }, { now: 200 }),
    ).toBe(false);
    expect(
      shouldSkipDuplicateInbound({ ...base, SessionKey: "agent:alpha:main" }, { now: 300 }),
    ).toBe(true);
  });
});

describe("createInboundDebouncer", () => {
  it("debounces and combines items", async () => {
    vi.useFakeTimers();
    const calls: Array<string[]> = [];

    const debouncer = createInboundDebouncer<{ key: string; id: string }>({
      debounceMs: 10,
      buildKey: (item) => item.key,
      onFlush: async (items) => {
        calls.push(items.map((entry) => entry.id));
      },
    });

    await debouncer.enqueue({ key: "a", id: "1" });
    await debouncer.enqueue({ key: "a", id: "2" });

    expect(calls).toEqual([]);
    await vi.advanceTimersByTimeAsync(10);
    expect(calls).toEqual([["1", "2"]]);

    vi.useRealTimers();
  });

  it("flushes buffered items before non-debounced item", async () => {
    vi.useFakeTimers();
    const calls: Array<string[]> = [];

    const debouncer = createInboundDebouncer<{ key: string; id: string; debounce: boolean }>({
      debounceMs: 50,
      buildKey: (item) => item.key,
      shouldDebounce: (item) => item.debounce,
      onFlush: async (items) => {
        calls.push(items.map((entry) => entry.id));
      },
    });

    await debouncer.enqueue({ key: "a", id: "1", debounce: true });
    await debouncer.enqueue({ key: "a", id: "2", debounce: false });

    expect(calls).toEqual([["1"], ["2"]]);

    vi.useRealTimers();
  });
});

describe("initSessionState BodyStripped", () => {
  it("prefers BodyForAgent over Body for group chats", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "marv-sender-meta-"));
    const storePath = path.join(root, "sessions.json");
    const cfg = { session: { store: storePath } } as MarvConfig;

    const result = await initSessionState({
      ctx: {
        Body: "[WhatsApp 123@g.us] ping",
        BodyForAgent: "ping",
        ChatType: "group",
        SenderName: "Bob",
        SenderE164: "+222",
        SenderId: "222@s.whatsapp.net",
        SessionKey: "agent:main:whatsapp:group:123@g.us",
      },
      cfg,
      commandAuthorized: true,
    });

    expect(result.sessionCtx.BodyStripped).toBe("ping");
  });

  it("prefers BodyForAgent over Body for direct chats", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "marv-sender-meta-direct-"));
    const storePath = path.join(root, "sessions.json");
    const cfg = { session: { store: storePath } } as MarvConfig;

    const result = await initSessionState({
      ctx: {
        Body: "[WhatsApp +1] ping",
        BodyForAgent: "ping",
        ChatType: "direct",
        SenderName: "Bob",
        SenderE164: "+222",
        SessionKey: "agent:main:whatsapp:dm:+222",
      },
      cfg,
      commandAuthorized: true,
    });

    expect(result.sessionCtx.BodyStripped).toBe("ping");
  });
});

describe("mention helpers", () => {
  it("builds regexes and skips invalid patterns", () => {
    const regexes = buildMentionRegexes({
      messages: {
        groupChat: { mentionPatterns: ["\\bmarv\\b", "(invalid"] },
      },
    });
    expect(regexes).toHaveLength(1);
    expect(regexes[0]?.test("marv")).toBe(true);
  });

  it("normalizes zero-width characters", () => {
    expect(normalizeMentionText("open\u200bclaw")).toBe("marv");
  });

  it("matches patterns case-insensitively", () => {
    const regexes = buildMentionRegexes({
      messages: { groupChat: { mentionPatterns: ["\\bmarv\\b"] } },
    });
    expect(matchesMentionPatterns("MARV: hi", regexes)).toBe(true);
  });

  it("uses per-agent mention patterns when configured", () => {
    const regexes = buildMentionRegexes(
      {
        messages: {
          groupChat: { mentionPatterns: ["\\bglobal\\b"] },
        },
        agents: {
          defaults: {
            groupChat: { mentionPatterns: ["\\bworkbot\\b"] },
          },
        },
      },
      "main",
    );
    expect(matchesMentionPatterns("workbot: hi", regexes)).toBe(true);
    expect(matchesMentionPatterns("global: hi", regexes)).toBe(false);
  });
});

describe("resolveGroupRequireMention", () => {
  it("respects Discord guild/channel requireMention settings", () => {
    const cfg: MarvConfig = {
      channels: {
        discord: {
          guilds: {
            "145": {
              requireMention: false,
              channels: {
                general: { allow: true },
              },
            },
          },
        },
      },
    };
    const ctx: SessionTemplateContext = {
      Provider: "discord",
      From: "discord:group:123",
      GroupChannel: "#general",
      GroupSpace: "145",
    };
    const groupResolution: GroupKeyResolution = {
      key: "discord:group:123",
      channel: "discord",
      id: "123",
      chatType: "group",
    };

    expect(resolveGroupRequireMention({ cfg, ctx, groupResolution })).toBe(false);
  });

  it("respects Slack channel requireMention settings", () => {
    const cfg: MarvConfig = {
      channels: {
        slack: {
          channels: {
            C123: { requireMention: false },
          },
        },
      },
    };
    const ctx: SessionTemplateContext = {
      Provider: "slack",
      From: "slack:channel:C123",
      GroupSubject: "#general",
    };
    const groupResolution: GroupKeyResolution = {
      key: "slack:group:C123",
      channel: "slack",
      id: "C123",
      chatType: "group",
    };

    expect(resolveGroupRequireMention({ cfg, ctx, groupResolution })).toBe(false);
  });
});

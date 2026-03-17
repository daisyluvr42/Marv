import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, test } from "vitest";
import type { MarvConfig } from "../config/config.js";
import type { SessionEntry } from "../config/sessions.js";
import {
  capArrayByJsonBytes,
  classifySessionKey,
  deriveSessionTitle,
  listSessionsFromStore,
  parseGroupKey,
  pruneLegacyStoreKeys,
  resolveGatewaySessionStoreTarget,
  resolveSessionModelRef,
  resolveSessionStoreKey,
} from "./session-utils.js";

describe("gateway session utils", () => {
  test("capArrayByJsonBytes trims from the front", () => {
    const res = capArrayByJsonBytes(["a", "b", "c"], 10);
    expect(res.items).toEqual(["b", "c"]);
  });

  test("parseGroupKey handles group keys", () => {
    expect(parseGroupKey("discord:group:dev")).toEqual({
      channel: "discord",
      kind: "group",
      id: "dev",
    });
    expect(parseGroupKey("agent:ops:discord:group:dev")).toEqual({
      channel: "discord",
      kind: "group",
      id: "dev",
    });
    expect(parseGroupKey("foo:bar")).toBeNull();
  });

  test("classifySessionKey respects chat type + prefixes", () => {
    expect(classifySessionKey("global")).toBe("global");
    expect(classifySessionKey("unknown")).toBe("unknown");
    expect(classifySessionKey("discord:group:dev")).toBe("group");
    expect(classifySessionKey("main")).toBe("direct");
    const entry = { chatType: "group" } as SessionEntry;
    expect(classifySessionKey("main", entry)).toBe("group");
  });

  test("resolveSessionStoreKey maps main aliases to default agent main", () => {
    const cfg = {
      session: { mainKey: "work" },
    } as MarvConfig;
    expect(resolveSessionStoreKey({ cfg, sessionKey: "main" })).toBe("agent:main:work");
    expect(resolveSessionStoreKey({ cfg, sessionKey: "work" })).toBe("agent:main:work");
    expect(resolveSessionStoreKey({ cfg, sessionKey: "agent:main:main" })).toBe("agent:main:work");
    // Mixed-case main alias must also resolve to the configured mainKey (idempotent)
    expect(resolveSessionStoreKey({ cfg, sessionKey: "agent:main:MAIN" })).toBe("agent:main:work");
    expect(resolveSessionStoreKey({ cfg, sessionKey: "MAIN" })).toBe("agent:main:work");
  });

  test("resolveSessionStoreKey canonicalizes bare keys to default agent", () => {
    const cfg = {
      session: { mainKey: "main" },
    } as MarvConfig;
    expect(resolveSessionStoreKey({ cfg, sessionKey: "discord:group:123" })).toBe(
      "agent:main:discord:group:123",
    );
    expect(resolveSessionStoreKey({ cfg, sessionKey: "agent:alpha:main" })).toBe(
      "agent:alpha:main",
    );
  });

  test("resolveSessionStoreKey always uses main agent when agents.list is absent", () => {
    const cfg = {
      session: { mainKey: "main" },
    } as MarvConfig;
    expect(resolveSessionStoreKey({ cfg, sessionKey: "main" })).toBe("agent:main:main");
    expect(resolveSessionStoreKey({ cfg, sessionKey: "discord:group:123" })).toBe(
      "agent:main:discord:group:123",
    );
  });

  test("resolveSessionStoreKey falls back to main when agents.list is missing", () => {
    const cfg = {
      session: { mainKey: "work" },
    } as MarvConfig;
    expect(resolveSessionStoreKey({ cfg, sessionKey: "main" })).toBe("agent:main:work");
    expect(resolveSessionStoreKey({ cfg, sessionKey: "thread-1" })).toBe("agent:main:thread-1");
  });

  test("resolveSessionStoreKey normalizes session key casing", () => {
    const cfg = {
      session: { mainKey: "main" },
    } as MarvConfig;
    // Bare keys with different casing must resolve to the same canonical key
    expect(resolveSessionStoreKey({ cfg, sessionKey: "CoP" })).toBe(
      resolveSessionStoreKey({ cfg, sessionKey: "cop" }),
    );
    expect(resolveSessionStoreKey({ cfg, sessionKey: "MySession" })).toBe("agent:main:mysession");
    // Prefixed agent keys with mixed-case rest must also normalize
    expect(resolveSessionStoreKey({ cfg, sessionKey: "agent:ops:CoP" })).toBe("agent:ops:cop");
    expect(resolveSessionStoreKey({ cfg, sessionKey: "agent:alpha:MySession" })).toBe(
      "agent:alpha:mysession",
    );
  });

  test("resolveSessionStoreKey honors global scope", () => {
    const cfg = {
      session: { scope: "global", mainKey: "work" },
    } as MarvConfig;
    expect(resolveSessionStoreKey({ cfg, sessionKey: "main" })).toBe("global");
    const target = resolveGatewaySessionStoreTarget({ cfg, key: "main" });
    expect(target.canonicalKey).toBe("global");
    expect(target.agentId).toBe("main");
  });

  test("resolveGatewaySessionStoreTarget uses canonical key for main alias", () => {
    const storeTemplate = path.join(
      os.tmpdir(),
      "marv-session-utils",
      "{agentId}",
      "sessions.json",
    );
    const cfg = {
      session: { mainKey: "main", store: storeTemplate },
    } as MarvConfig;
    const target = resolveGatewaySessionStoreTarget({ cfg, key: "main" });
    expect(target.canonicalKey).toBe("agent:main:main");
    expect(target.storeKeys).toEqual(expect.arrayContaining(["agent:main:main", "main"]));
    expect(target.storePath).toBe(path.resolve(storeTemplate.replace("{agentId}", "main")));
  });

  test("resolveGatewaySessionStoreTarget includes legacy mixed-case store key", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "session-utils-case-"));
    const storePath = path.join(dir, "sessions.json");
    // Simulate a legacy store with a mixed-case key
    fs.writeFileSync(
      storePath,
      JSON.stringify({ "agent:main:MySession": { sessionId: "s1", updatedAt: 1 } }),
      "utf8",
    );
    const cfg = {
      session: { mainKey: "main", store: storePath },
    } as MarvConfig;
    // Client passes the lowercased canonical key (as returned by sessions.list)
    const target = resolveGatewaySessionStoreTarget({ cfg, key: "agent:main:mysession" });
    expect(target.canonicalKey).toBe("agent:main:mysession");
    // storeKeys must include the legacy mixed-case key from the on-disk store
    expect(target.storeKeys).toEqual(
      expect.arrayContaining(["agent:main:mysession", "agent:main:MySession"]),
    );
    // The legacy key must resolve to the actual entry in the store
    const store = JSON.parse(fs.readFileSync(storePath, "utf8"));
    const found = target.storeKeys.some((k) => Boolean(store[k]));
    expect(found).toBe(true);
  });

  test("resolveGatewaySessionStoreTarget includes all case-variant duplicate keys", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "session-utils-dupes-"));
    const storePath = path.join(dir, "sessions.json");
    // Simulate a store with both canonical and legacy mixed-case entries
    fs.writeFileSync(
      storePath,
      JSON.stringify({
        "agent:main:mysession": { sessionId: "s-lower", updatedAt: 2 },
        "agent:main:MySession": { sessionId: "s-mixed", updatedAt: 1 },
      }),
      "utf8",
    );
    const cfg = {
      session: { mainKey: "main", store: storePath },
    } as MarvConfig;
    const target = resolveGatewaySessionStoreTarget({ cfg, key: "agent:main:mysession" });
    // storeKeys must include BOTH variants so delete/reset/patch can clean up all duplicates
    expect(target.storeKeys).toEqual(
      expect.arrayContaining(["agent:main:mysession", "agent:main:MySession"]),
    );
  });

  test("resolveGatewaySessionStoreTarget finds legacy main alias key when mainKey is customized", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "session-utils-alias-"));
    const storePath = path.join(dir, "sessions.json");
    // Legacy store has entry under "agent:main:MAIN" but mainKey is "work"
    fs.writeFileSync(
      storePath,
      JSON.stringify({ "agent:main:MAIN": { sessionId: "s1", updatedAt: 1 } }),
      "utf8",
    );
    const cfg = {
      session: { mainKey: "work", store: storePath },
    } as MarvConfig;
    const target = resolveGatewaySessionStoreTarget({ cfg, key: "agent:main:main" });
    expect(target.canonicalKey).toBe("agent:main:work");
    // storeKeys must include the legacy mixed-case alias key
    expect(target.storeKeys).toEqual(expect.arrayContaining(["agent:main:MAIN"]));
  });

  test("pruneLegacyStoreKeys removes alias and case-variant ghost keys", () => {
    const store: Record<string, unknown> = {
      "agent:main:work": { sessionId: "canonical", updatedAt: 3 },
      "agent:main:MAIN": { sessionId: "legacy-upper", updatedAt: 1 },
      "agent:main:Main": { sessionId: "legacy-mixed", updatedAt: 2 },
      "agent:main:main": { sessionId: "legacy-lower", updatedAt: 4 },
    };
    pruneLegacyStoreKeys({
      store,
      canonicalKey: "agent:main:work",
      candidates: ["agent:main:work", "agent:main:main"],
    });
    expect(Object.keys(store).toSorted()).toEqual(["agent:main:work"]);
  });
});

describe("resolveSessionModelRef", () => {
  test("prefers explicit override over the last runtime model/provider", () => {
    const cfg = {
      agents: {
        defaults: {
          model: { primary: "anthropic/claude-opus-4-6" },
        },
      },
    } as MarvConfig;

    const resolved = resolveSessionModelRef(cfg, {
      sessionId: "s1",
      updatedAt: Date.now(),
      modelProvider: "openai-codex",
      model: "gpt-5.3-codex",
      modelOverride: "claude-opus-4-6",
      providerOverride: "anthropic",
    });

    expect(resolved).toEqual({ provider: "anthropic", model: "claude-opus-4-6" });
  });

  test("falls back to override when runtime model is not recorded yet", () => {
    const cfg = {
      agents: {
        defaults: {
          model: { primary: "anthropic/claude-opus-4-6" },
        },
      },
    } as MarvConfig;

    const resolved = resolveSessionModelRef(cfg, {
      sessionId: "s2",
      updatedAt: Date.now(),
      modelOverride: "openai-codex/gpt-5.3-codex",
    });

    expect(resolved).toEqual({ provider: "openai-codex", model: "gpt-5.3-codex" });
  });

  test("uses the last runtime model in auto mode when available", () => {
    const cfg = {
      agents: {
        defaults: {
          model: { primary: "anthropic/claude-opus-4-6" },
        },
      },
    } as MarvConfig;

    const resolved = resolveSessionModelRef(cfg, {
      sessionId: "s3",
      updatedAt: Date.now(),
      selectionMode: "auto",
      modelProvider: "openai-codex",
      model: "gpt-5.3-codex",
    });

    expect(resolved).toEqual({ provider: "openai-codex", model: "gpt-5.3-codex" });
  });

  test("prefers explicit manual selection state over legacy runtime fields", () => {
    const cfg = {
      agents: {
        defaults: {
          model: { primary: "anthropic/claude-opus-4-6" },
        },
      },
    } as MarvConfig;

    const resolved = resolveSessionModelRef(cfg, {
      sessionId: "s4",
      updatedAt: Date.now(),
      modelProvider: "openai-codex",
      model: "gpt-5.3-codex",
      selectionMode: "manual",
      manualModelRef: "openai/gpt-4o",
    });

    expect(resolved).toEqual({ provider: "openai", model: "gpt-4o" });
  });
});

describe("deriveSessionTitle", () => {
  test("returns undefined for undefined entry", () => {
    expect(deriveSessionTitle(undefined)).toBeUndefined();
  });

  test("prefers displayName when set", () => {
    const entry = {
      sessionId: "abc123",
      updatedAt: Date.now(),
      displayName: "My Custom Session",
      subject: "Group Chat",
    } as SessionEntry;
    expect(deriveSessionTitle(entry)).toBe("My Custom Session");
  });

  test("falls back to subject when displayName is missing", () => {
    const entry = {
      sessionId: "abc123",
      updatedAt: Date.now(),
      subject: "Dev Team Chat",
    } as SessionEntry;
    expect(deriveSessionTitle(entry)).toBe("Dev Team Chat");
  });

  test("uses first user message when displayName and subject missing", () => {
    const entry = {
      sessionId: "abc123",
      updatedAt: Date.now(),
    } as SessionEntry;
    expect(deriveSessionTitle(entry, "Hello, how are you?")).toBe("Hello, how are you?");
  });

  test("truncates long first user message to 60 chars with ellipsis", () => {
    const entry = {
      sessionId: "abc123",
      updatedAt: Date.now(),
    } as SessionEntry;
    const longMsg =
      "This is a very long message that exceeds sixty characters and should be truncated appropriately";
    const result = deriveSessionTitle(entry, longMsg);
    expect(result).toBeDefined();
    expect(result!.length).toBeLessThanOrEqual(60);
    expect(result!.endsWith("…")).toBe(true);
  });

  test("truncates at word boundary when possible", () => {
    const entry = {
      sessionId: "abc123",
      updatedAt: Date.now(),
    } as SessionEntry;
    const longMsg = "This message has many words and should be truncated at a word boundary nicely";
    const result = deriveSessionTitle(entry, longMsg);
    expect(result).toBeDefined();
    expect(result!.endsWith("…")).toBe(true);
    expect(result!.includes("  ")).toBe(false);
  });

  test("falls back to sessionId prefix with date", () => {
    const entry = {
      sessionId: "abcd1234-5678-90ef-ghij-klmnopqrstuv",
      updatedAt: new Date("2024-03-15T10:30:00Z").getTime(),
    } as SessionEntry;
    const result = deriveSessionTitle(entry);
    expect(result).toBe("abcd1234 (2024-03-15)");
  });

  test("falls back to sessionId prefix without date when updatedAt missing", () => {
    const entry = {
      sessionId: "abcd1234-5678-90ef-ghij-klmnopqrstuv",
      updatedAt: 0,
    } as SessionEntry;
    const result = deriveSessionTitle(entry);
    expect(result).toBe("abcd1234");
  });

  test("trims whitespace from displayName", () => {
    const entry = {
      sessionId: "abc123",
      updatedAt: Date.now(),
      displayName: "  Padded Name  ",
    } as SessionEntry;
    expect(deriveSessionTitle(entry)).toBe("Padded Name");
  });

  test("ignores empty displayName and falls through", () => {
    const entry = {
      sessionId: "abc123",
      updatedAt: Date.now(),
      displayName: "   ",
      subject: "Actual Subject",
    } as SessionEntry;
    expect(deriveSessionTitle(entry)).toBe("Actual Subject");
  });
});

describe("listSessionsFromStore search", () => {
  const baseCfg = {
    session: { mainKey: "main" },
    agents: { list: [{ id: "main", default: true }] },
  } as MarvConfig;

  const makeStore = (): Record<string, SessionEntry> => ({
    "agent:main:work-project": {
      sessionId: "sess-work-1",
      updatedAt: Date.now(),
      displayName: "Work Project Alpha",
      label: "work",
    } as SessionEntry,
    "agent:main:personal-chat": {
      sessionId: "sess-personal-1",
      updatedAt: Date.now() - 1000,
      displayName: "Personal Chat",
      subject: "Family Reunion Planning",
    } as SessionEntry,
    "agent:main:discord:group:dev-team": {
      sessionId: "sess-discord-1",
      updatedAt: Date.now() - 2000,
      label: "discord",
      subject: "Dev Team Discussion",
    } as SessionEntry,
  });

  test("returns all sessions when search is empty", () => {
    const store = makeStore();
    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: { search: "" },
    });
    expect(result.sessions.length).toBe(3);
  });

  test("returns all sessions when search is undefined", () => {
    const store = makeStore();
    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: {},
    });
    expect(result.sessions.length).toBe(3);
  });

  test("filters by displayName case-insensitively", () => {
    const store = makeStore();
    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: { search: "WORK PROJECT" },
    });
    expect(result.sessions.length).toBe(1);
    expect(result.sessions[0].displayName).toBe("Work Project Alpha");
  });

  test("filters by subject", () => {
    const store = makeStore();
    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: { search: "reunion" },
    });
    expect(result.sessions.length).toBe(1);
    expect(result.sessions[0].subject).toBe("Family Reunion Planning");
  });

  test("filters by label", () => {
    const store = makeStore();
    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: { search: "discord" },
    });
    expect(result.sessions.length).toBe(1);
    expect(result.sessions[0].label).toBe("discord");
  });

  test("filters by sessionId", () => {
    const store = makeStore();
    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: { search: "sess-personal" },
    });
    expect(result.sessions.length).toBe(1);
    expect(result.sessions[0].sessionId).toBe("sess-personal-1");
  });

  test("filters by key", () => {
    const store = makeStore();
    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: { search: "dev-team" },
    });
    expect(result.sessions.length).toBe(1);
    expect(result.sessions[0].key).toBe("agent:main:discord:group:dev-team");
  });

  test("returns empty array when no matches", () => {
    const store = makeStore();
    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: { search: "nonexistent-term" },
    });
    expect(result.sessions.length).toBe(0);
  });

  test("matches partial strings", () => {
    const store = makeStore();
    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: { search: "alpha" },
    });
    expect(result.sessions.length).toBe(1);
    expect(result.sessions[0].displayName).toBe("Work Project Alpha");
  });

  test("trims whitespace from search query", () => {
    const store = makeStore();
    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: { search: "  personal  " },
    });
    expect(result.sessions.length).toBe(1);
  });

  test("hides cron run alias session keys from sessions list", () => {
    const now = Date.now();
    const store: Record<string, SessionEntry> = {
      "agent:main:cron:job-1": {
        sessionId: "run-abc",
        updatedAt: now,
        label: "Cron: job-1",
      } as SessionEntry,
      "agent:main:cron:job-1:run:run-abc": {
        sessionId: "run-abc",
        updatedAt: now,
        label: "Cron: job-1",
      } as SessionEntry,
    };

    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: {},
    });

    expect(result.sessions.map((session) => session.key)).toEqual(["agent:main:cron:job-1"]);
  });

  test("exposes unknown totals when freshness is stale or missing", () => {
    const now = Date.now();
    const store: Record<string, SessionEntry> = {
      "agent:main:fresh": {
        sessionId: "sess-fresh",
        updatedAt: now,
        totalTokens: 1200,
        totalTokensFresh: true,
      } as SessionEntry,
      "agent:main:stale": {
        sessionId: "sess-stale",
        updatedAt: now - 1000,
        totalTokens: 2200,
        totalTokensFresh: false,
      } as SessionEntry,
      "agent:main:missing": {
        sessionId: "sess-missing",
        updatedAt: now - 2000,
        inputTokens: 100,
        outputTokens: 200,
      } as SessionEntry,
    };

    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: {},
    });

    const fresh = result.sessions.find((row) => row.key === "agent:main:fresh");
    const stale = result.sessions.find((row) => row.key === "agent:main:stale");
    const missing = result.sessions.find((row) => row.key === "agent:main:missing");
    expect(fresh?.totalTokens).toBe(1200);
    expect(fresh?.totalTokensFresh).toBe(true);
    expect(stale?.totalTokens).toBeUndefined();
    expect(stale?.totalTokensFresh).toBe(false);
    expect(missing?.totalTokens).toBeUndefined();
    expect(missing?.totalTokensFresh).toBe(false);
  });

  test("keeps raw override fields alongside the resolved footer model", () => {
    const store: Record<string, SessionEntry> = {
      "agent:main:main": {
        sessionId: "sess-main",
        updatedAt: Date.now(),
        modelProvider: "openai-codex",
        model: "gpt-5.3-codex",
        providerOverride: "anthropic",
        modelOverride: "claude-opus-4-6",
      } as SessionEntry,
    };

    const result = listSessionsFromStore({
      cfg: baseCfg,
      storePath: "/tmp/sessions.json",
      store,
      opts: {},
    });

    expect(result.sessions[0]).toMatchObject({
      key: "agent:main:main",
      providerOverride: "anthropic",
      modelOverride: "claude-opus-4-6",
      modelProvider: "anthropic",
      model: "claude-opus-4-6",
    });
  });
});

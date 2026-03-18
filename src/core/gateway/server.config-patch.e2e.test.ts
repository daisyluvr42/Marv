import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  connectOk,
  installGatewayTestHooks,
  rpcReq,
  startServerWithClient,
  testState,
  writeSessionStore,
} from "./test-helpers.js";

installGatewayTestHooks({ scope: "suite" });

let server: Awaited<ReturnType<typeof startServerWithClient>>["server"];
let ws: Awaited<ReturnType<typeof startServerWithClient>>["ws"];

beforeAll(async () => {
  const started = await startServerWithClient(undefined, { controlUiEnabled: true });
  server = started.server;
  ws = started.ws;
  await connectOk(ws);
});

afterAll(async () => {
  ws.close();
  await server.close();
});

describe("gateway config methods", () => {
  const readConfigHash = async () => {
    const snapshotRes = await rpcReq<{ hash?: string }>(ws, "config.get", {});
    expect(snapshotRes.ok).toBe(true);
    expect(typeof snapshotRes.payload?.hash).toBe("string");
    return snapshotRes.payload?.hash ?? "";
  };

  it("returns a config snapshot", async () => {
    const res = await rpcReq<{ hash?: string; raw?: string }>(ws, "config.get", {});
    expect(res.ok).toBe(true);
    const payload = res.payload ?? {};
    expect(typeof payload.raw === "string" || typeof payload.hash === "string").toBe(true);
  });

  it("rejects config.patch when raw is not an object", async () => {
    const res = await rpcReq<{ ok?: boolean }>(ws, "config.patch", {
      raw: "[]",
    });
    expect(res.ok).toBe(false);
    expect(res.error?.message ?? "").toContain("raw must be an object");
  });

  it("rejects config.set when top-level agents.list is provided", async () => {
    const setRes = await rpcReq<{ ok?: boolean }>(ws, "config.set", {
      raw: JSON.stringify({
        agents: {
          list: [{ id: "primary", default: true, workspace: "/tmp/primary" }],
        },
      }),
    });
    expect(setRes.ok).toBe(false);
    expect(setRes.error?.message ?? "").toContain("invalid config");
    expect(
      JSON.stringify((setRes.error as { details?: unknown } | undefined)?.details ?? {}),
    ).toContain("agents.list");
  });

  it("rejects config.patch when legacy multi-agent fields are targeted", async () => {
    const baseHash = await readConfigHash();

    const patchRes = await rpcReq<{ ok?: boolean }>(ws, "config.patch", {
      baseHash,
      raw: JSON.stringify({
        agents: {
          list: [{ id: "primary", workspace: "/tmp/primary-updated" }],
        },
      }),
    });
    expect(patchRes.ok).toBe(false);
    expect(patchRes.error?.message ?? "").toContain("legacy multi-agent config");
  });

  it("rejects bindings patches without mutating persisted config", async () => {
    const beforeHash = await readConfigHash();

    const patchRes = await rpcReq<{ ok?: boolean }>(ws, "config.patch", {
      baseHash: beforeHash,
      raw: JSON.stringify({
        bindings: [{ agentId: "main", match: { channel: "telegram" } }],
      }),
    });
    expect(patchRes.ok).toBe(false);
    expect(patchRes.error?.message ?? "").toContain("legacy multi-agent config");

    const afterHash = await readConfigHash();
    expect(afterHash).toBe(beforeHash);
  });

  it("supports semantic patch lifecycle and ledger query", async () => {
    const proposeRes = await rpcReq<{
      proposalId?: string;
      status?: string;
      patch?: Record<string, unknown>;
    }>(ws, "config.patches.propose", {
      naturalLanguage: "请更简洁一点",
      scopeType: "global",
      scopeId: "gateway",
    });
    expect(proposeRes.ok).toBe(true);
    const proposalId = proposeRes.payload?.proposalId ?? "";
    expect(proposalId).toMatch(/^pp_/);
    expect(proposeRes.payload?.status).toBe("open");

    const commitRes = await rpcReq<{
      status?: string;
      revision?: string;
    }>(ws, "config.patches.commit", {
      proposalId,
    });
    expect(commitRes.ok).toBe(true);
    expect(commitRes.payload?.status).toBe("committed");
    const revision = commitRes.payload?.revision ?? "";
    expect(revision).toMatch(/^rev_/);

    const listRes = await rpcReq<{
      revisions?: Array<{ revision?: string }>;
    }>(ws, "config.revisions.list", {
      scopeType: "global",
      scopeId: "gateway",
      limit: 50,
    });
    expect(listRes.ok).toBe(true);
    const revisions = listRes.payload?.revisions ?? [];
    expect(revisions.some((entry) => entry.revision === revision)).toBe(true);

    const rollbackRes = await rpcReq<{
      rolledBack?: string;
      rollbackRevision?: string;
    }>(ws, "config.revisions.rollback", {
      revision,
    });
    expect(rollbackRes.ok).toBe(true);
    expect(rollbackRes.payload?.rolledBack).toBe(revision);
    expect(rollbackRes.payload?.rollbackRevision ?? "").toMatch(/^rev_/);

    const ledgerRes = await rpcReq<{
      events?: Array<{ type?: string }>;
    }>(ws, "ledger.events.query", {
      conversationId: "config:global:gateway",
      limit: 100,
    });
    expect(ledgerRes.ok).toBe(true);
    const eventTypes = (ledgerRes.payload?.events ?? []).map((entry) => entry.type);
    expect(eventTypes).toContain("PatchProposedEvent");
    expect(eventTypes).toContain("PatchCommittedEvent");
    expect(eventTypes).toContain("PatchRolledBackEvent");
  });
});

describe("gateway server sessions", () => {
  it("only serves main-agent sessions in the main-only architecture", async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-sessions-agents-"));
    testState.sessionConfig = {
      store: path.join(dir, "{agentId}", "sessions.json"),
    };
    testState.agentsConfig = {
      list: [{ id: "home", default: true }, { id: "work" }],
    };
    const mainDir = path.join(dir, "main");
    const workDir = path.join(dir, "work");
    await fs.mkdir(mainDir, { recursive: true });
    await fs.mkdir(workDir, { recursive: true });
    await writeSessionStore({
      storePath: path.join(mainDir, "sessions.json"),
      agentId: "main",
      entries: {
        main: {
          sessionId: "sess-main-main",
          updatedAt: Date.now(),
        },
        "discord:group:dev": {
          sessionId: "sess-main-group",
          updatedAt: Date.now() - 1000,
        },
      },
    });
    await writeSessionStore({
      storePath: path.join(workDir, "sessions.json"),
      agentId: "work",
      entries: {
        main: {
          sessionId: "sess-work-main",
          updatedAt: Date.now(),
        },
      },
    });

    const mainSessions = await rpcReq<{
      sessions: Array<{ key: string }>;
    }>(ws, "sessions.list", {
      includeGlobal: false,
      includeUnknown: false,
      agentId: "main",
    });
    expect(mainSessions.ok).toBe(true);
    expect(mainSessions.payload?.sessions.map((s) => s.key).toSorted()).toEqual([
      "agent:main:discord:group:dev",
      "agent:main:main",
    ]);

    const legacySessions = await rpcReq<{
      sessions: Array<{ key: string }>;
    }>(ws, "sessions.list", {
      includeGlobal: false,
      includeUnknown: false,
      agentId: "work",
    });
    expect(legacySessions.ok).toBe(true);
    expect(legacySessions.payload?.sessions).toEqual([]);
  });

  it("resolves and patches main alias to the durable main agent key", async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-sessions-"));
    const storePath = path.join(dir, "sessions.json");
    testState.sessionStorePath = storePath;
    testState.agentsConfig = undefined;
    testState.sessionConfig = { mainKey: "work" };

    await writeSessionStore({
      storePath,
      agentId: "main",
      mainKey: "work",
      entries: {
        main: {
          sessionId: "sess-main-main",
          updatedAt: Date.now(),
        },
      },
    });

    const resolved = await rpcReq<{ ok: true; key: string }>(ws, "sessions.resolve", {
      key: "main",
    });
    expect(resolved.ok).toBe(true);
    expect(resolved.payload?.key).toBe("agent:main:work");

    const patched = await rpcReq<{ ok: true; key: string }>(ws, "sessions.patch", {
      key: "main",
      thinkingLevel: "medium",
    });
    expect(patched.ok).toBe(true);
    expect(patched.payload?.key).toBe("agent:main:work");

    const stored = JSON.parse(await fs.readFile(storePath, "utf-8")) as Record<
      string,
      { thinkingLevel?: string }
    >;
    expect(stored["agent:main:work"]?.thinkingLevel).toBe("medium");
    expect(stored.main).toBeUndefined();
  });
});

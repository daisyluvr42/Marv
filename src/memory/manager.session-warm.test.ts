import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getMemorySearchManager, type MemoryIndexManager } from "./index.js";
import "./test-runtime-mocks.js";

vi.mock("./embeddings/embeddings.js", () => {
  const embedText = (_text: string) => [1, 0];
  return {
    createEmbeddingProvider: async (options: { model?: string }) => ({
      requestedProvider: "openai",
      provider: {
        id: "mock",
        model: options.model ?? "mock-embed",
        embedQuery: async (text: string) => embedText(text),
        embedBatch: async (texts: string[]) => texts.map(embedText),
      },
    }),
  };
});

describe("memory manager session warm cache", () => {
  let fixtureRoot = "";
  let workspaceDir = "";
  let manager: MemoryIndexManager | null = null;

  beforeEach(async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-11T00:00:00.000Z"));
    fixtureRoot = await fs.mkdtemp(path.join(os.tmpdir(), "marv-session-warm-"));
    workspaceDir = path.join(fixtureRoot, "workspace");
    await fs.mkdir(path.join(workspaceDir, "memory"), { recursive: true });
  });

  afterEach(async () => {
    vi.useRealTimers();
    await manager?.close();
    manager = null;
    await fs.rm(fixtureRoot, { recursive: true, force: true });
  });

  it("reuses warm markers briefly and expires them after the ttl window", async () => {
    const storePath = path.join(workspaceDir, "index.sqlite");
    const cfg = {
      agents: {
        defaults: {
          workspace: workspaceDir,
          memorySearch: {
            provider: "openai" as const,
            model: "mock-embed",
            store: { path: storePath, vector: { enabled: false } },
            chunking: { tokens: 4000, overlap: 0 },
            sync: { watch: false, onSessionStart: true, onSearch: false },
            query: { minScore: 0, hybrid: { enabled: false } },
          },
        },
      },
    };

    const result = await getMemorySearchManager({ cfg, agentId: "main" });
    expect(result.manager).not.toBeNull();
    if (!result.manager) {
      throw new Error("manager missing");
    }
    manager = result.manager as MemoryIndexManager;

    const syncSpy = vi.spyOn(manager, "sync").mockResolvedValue(undefined);

    await manager.warmSession("agent:main:test");
    expect(syncSpy).toHaveBeenCalledTimes(1);

    await manager.warmSession("agent:main:test");
    expect(syncSpy).toHaveBeenCalledTimes(1);

    vi.setSystemTime(new Date("2026-03-11T00:30:01.000Z"));
    await manager.warmSession("agent:main:test");
    expect(syncSpy).toHaveBeenCalledTimes(2);
  });
});

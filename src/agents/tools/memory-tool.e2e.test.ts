import { beforeEach, describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";

let backend: "builtin" | "qmd" = "builtin";
let searchImpl: () => Promise<unknown[]> = async () => [
  {
    path: "MEMORY.md",
    startLine: 5,
    endLine: 7,
    score: 0.9,
    snippet: "@@ -5,3 @@\nAssistant: noted",
    source: "memory" as const,
  },
];
let readFileImpl: () => Promise<string> = async () => "";
let soulSearchImpl: () => unknown[] = () => [];
let soulReadImpl: (itemId: string) => { id: string; content: string } | null = () => null;
let soulWriteImpl: (content: string) => { id: string } = () => ({ id: "mem_mock" });
let soulRefsImpl: (itemId: string) => string[] = () => [];

const stubManager = {
  search: vi.fn(async () => await searchImpl()),
  readFile: vi.fn(async () => await readFileImpl()),
  status: () => ({
    backend,
    files: 1,
    chunks: 1,
    dirty: false,
    workspaceDir: "/workspace",
    dbPath: "/workspace/.memory/index.sqlite",
    provider: "builtin",
    model: "builtin",
    requestedProvider: "builtin",
    sources: ["memory" as const],
    sourceCounts: [{ source: "memory" as const, files: 1, chunks: 1 }],
  }),
  sync: vi.fn(),
  probeVectorAvailability: vi.fn(async () => true),
  close: vi.fn(),
};

vi.mock("../../memory/index.js", () => {
  return {
    getMemorySearchManager: async () => ({ manager: stubManager }),
  };
});

vi.mock("../../memory/storage/soul-memory-store.js", () => {
  return {
    querySoulMemoryMulti: (..._args: unknown[]) => soulSearchImpl(),
    getSoulMemoryItem: ({ itemId }: { itemId: string }) => soulReadImpl(itemId),
    listSoulMemoryReferences: ({ itemId }: { itemId: string }) => soulRefsImpl(itemId),
    writeSoulMemory: ({ content }: { content: string }) => soulWriteImpl(content),
    buildSoulMemoryPath: (itemId: string) => `soul-memory/${itemId}`,
    parseSoulMemoryPath: (value: string) =>
      value.startsWith("soul-memory/") ? value.slice("soul-memory/".length) : null,
  };
});

import {
  createMemoryGetTool,
  createMemorySearchTool,
  createMemoryWriteTool,
} from "./memory-tool.js";

function asMarvConfig(config: Partial<MarvConfig>): MarvConfig {
  return config as MarvConfig;
}

beforeEach(() => {
  backend = "builtin";
  searchImpl = async () => [
    {
      path: "MEMORY.md",
      startLine: 5,
      endLine: 7,
      score: 0.9,
      snippet: "@@ -5,3 @@\nAssistant: noted",
      source: "memory" as const,
    },
  ];
  readFileImpl = async () => "";
  soulSearchImpl = () => [];
  soulReadImpl = () => null;
  soulWriteImpl = () => ({ id: "mem_mock" });
  soulRefsImpl = () => [];
  vi.clearAllMocks();
});

describe("memory search citations", () => {
  it("appends source information when citations are enabled", async () => {
    backend = "builtin";
    const cfg = asMarvConfig({
      memory: { citations: "on" },
      agents: { list: [{ id: "main", default: true }] },
    });
    const tool = createMemorySearchTool({ config: cfg });
    if (!tool) {
      throw new Error("tool missing");
    }
    const result = await tool.execute("call_citations_on", { query: "notes" });
    const details = result.details as { results: Array<{ snippet: string; citation?: string }> };
    expect(details.results[0]?.snippet).toMatch(/Source: MEMORY.md#L5-L7/);
    expect(details.results[0]?.citation).toBe("MEMORY.md#L5-L7");
  });

  it("leaves snippet untouched when citations are off", async () => {
    backend = "builtin";
    const cfg = asMarvConfig({
      memory: { citations: "off" },
      agents: { list: [{ id: "main", default: true }] },
    });
    const tool = createMemorySearchTool({ config: cfg });
    if (!tool) {
      throw new Error("tool missing");
    }
    const result = await tool.execute("call_citations_off", { query: "notes" });
    const details = result.details as { results: Array<{ snippet: string; citation?: string }> };
    expect(details.results[0]?.snippet).not.toMatch(/Source:/);
    expect(details.results[0]?.citation).toBeUndefined();
  });

  it("clamps decorated snippets to qmd injected budget", async () => {
    backend = "qmd";
    const cfg = asMarvConfig({
      memory: { citations: "on", backend: "qmd", qmd: { limits: { maxInjectedChars: 20 } } },
      agents: { list: [{ id: "main", default: true }] },
    });
    const tool = createMemorySearchTool({ config: cfg });
    if (!tool) {
      throw new Error("tool missing");
    }
    const result = await tool.execute("call_citations_qmd", { query: "notes" });
    const details = result.details as { results: Array<{ snippet: string; citation?: string }> };
    expect(details.results[0]?.snippet.length).toBeLessThanOrEqual(20);
  });

  it("honors auto mode for direct chats", async () => {
    backend = "builtin";
    const cfg = asMarvConfig({
      memory: { citations: "auto" },
      agents: { list: [{ id: "main", default: true }] },
    });
    const tool = createMemorySearchTool({
      config: cfg,
      agentSessionKey: "agent:main:discord:dm:u123",
    });
    if (!tool) {
      throw new Error("tool missing");
    }
    const result = await tool.execute("auto_mode_direct", { query: "notes" });
    const details = result.details as { results: Array<{ snippet: string }> };
    expect(details.results[0]?.snippet).toMatch(/Source:/);
  });

  it("suppresses citations for auto mode in group chats", async () => {
    backend = "builtin";
    const cfg = asMarvConfig({
      memory: { citations: "auto" },
      agents: { list: [{ id: "main", default: true }] },
    });
    const tool = createMemorySearchTool({
      config: cfg,
      agentSessionKey: "agent:main:discord:group:c123",
    });
    if (!tool) {
      throw new Error("tool missing");
    }
    const result = await tool.execute("auto_mode_group", { query: "notes" });
    const details = result.details as { results: Array<{ snippet: string }> };
    expect(details.results[0]?.snippet).not.toMatch(/Source:/);
  });
});

describe("memory tools", () => {
  it("skips memory search on small-talk when precheck is enabled", async () => {
    const soulSearchSpy = vi.fn(() => []);
    soulSearchImpl = soulSearchSpy;
    const cfg = asMarvConfig({
      agents: {
        defaults: {
          memorySearch: {
            query: {
              precheck: {
                enabled: true,
              },
            },
          },
        },
        list: [{ id: "main", default: true }],
      },
    });
    const tool = createMemorySearchTool({ config: cfg });
    if (!tool) {
      throw new Error("tool missing");
    }

    const result = await tool.execute("call_precheck_skip", { query: "hi" });
    expect(result.details).toMatchObject({
      results: [],
      skipped: true,
      reason: "small-talk",
      mode: "precheck_skip",
    });
    expect(stubManager.search).not.toHaveBeenCalled();
    expect(soulSearchSpy).not.toHaveBeenCalled();
  });

  it("rewrites memory query before searching when precheck rewrite is enabled", async () => {
    const soulSearchSpy = vi.fn(() => []);
    soulSearchImpl = soulSearchSpy;
    searchImpl = async () => [];
    const cfg = asMarvConfig({
      agents: {
        defaults: {
          memorySearch: {
            query: {
              precheck: {
                enabled: true,
                rewrite: true,
              },
            },
          },
        },
        list: [{ id: "main", default: true }],
      },
    });
    const tool = createMemorySearchTool({ config: cfg });
    if (!tool) {
      throw new Error("tool missing");
    }

    const result = await tool.execute("call_precheck_rewrite", {
      query: "can you remember deployment checklist?",
    });
    expect(stubManager.search).toHaveBeenCalledWith(
      "deployment checklist",
      expect.objectContaining({
        maxResults: 6,
      }),
    );
    const details = result.details as { rewrittenQuery?: string };
    expect(details.rewrittenQuery).toBe("deployment checklist");
    expect(soulSearchSpy).toHaveBeenCalledTimes(1);
  });

  it("does not throw when memory_search fails (e.g. embeddings 429)", async () => {
    searchImpl = async () => {
      throw new Error("openai embeddings failed: 429 insufficient_quota");
    };

    const cfg = { agents: { list: [{ id: "main", default: true }] } };
    const tool = createMemorySearchTool({ config: cfg });
    expect(tool).not.toBeNull();
    if (!tool) {
      throw new Error("tool missing");
    }

    const result = await tool.execute("call_1", { query: "hello" });
    expect(result.details).toEqual({
      results: [],
      disabled: true,
      error: "openai embeddings failed: 429 insufficient_quota",
    });
  });

  it("does not throw when memory_get fails", async () => {
    readFileImpl = async () => {
      throw new Error("path required");
    };

    const cfg = { agents: { list: [{ id: "main", default: true }] } };
    const tool = createMemoryGetTool({ config: cfg });
    expect(tool).not.toBeNull();
    if (!tool) {
      throw new Error("tool missing");
    }

    const result = await tool.execute("call_2", { path: "memory/NOPE.md" });
    expect(result.details).toEqual({
      path: "memory/NOPE.md",
      text: "",
      disabled: true,
      error: "path required",
    });
  });

  it("returns references when reading a soul-memory path", async () => {
    soulReadImpl = () => ({ id: "mem_a", content: "follow-up linked note" });
    soulRefsImpl = () => ["mem_root"];

    const cfg = { agents: { list: [{ id: "main", default: true }] } };
    const tool = createMemoryGetTool({ config: cfg });
    if (!tool) {
      throw new Error("tool missing");
    }

    const result = await tool.execute("call_get_soul", { path: "soul-memory/mem_a" });
    expect(result.details).toEqual({
      path: "soul-memory/mem_a",
      text: "follow-up linked note",
      references: ["mem_root"],
    });
  });

  it("writes structured soul memory with memory_write", async () => {
    soulWriteImpl = () => ({ id: "mem_written" });
    soulRefsImpl = () => ["mem_anchor"];
    const cfg = { agents: { list: [{ id: "main", default: true }] } };
    const tool = createMemoryWriteTool({ config: cfg });
    expect(tool).not.toBeNull();
    if (!tool) {
      throw new Error("tool missing");
    }

    const result = await tool.execute("call_3", { content: "remember this decision" });
    expect(result.details).toMatchObject({
      ok: true,
      id: "mem_written",
      path: "soul-memory/mem_written",
      references: ["mem_anchor"],
    });
  });

  it("includes [ref:item_id] chain and salience metadata in soul search results", async () => {
    soulSearchImpl = () => [
      {
        id: "mem_linked",
        scopeType: "agent",
        scopeId: "main",
        kind: "note",
        content: "linked memory content",
        confidence: 0.9,
        tier: "P1",
        source: "manual_log",
        createdAt: 100,
        lastAccessedAt: 100,
        reinforcementCount: 3,
        lastReinforcedAt: 100,
        score: 0.8,
        vectorScore: 0.8,
        lexicalScore: 0.8,
        bm25Score: 0.6,
        rrfScore: 0.6,
        graphScore: 0.1,
        clusterScore: 0.2,
        relevanceScore: 0.8,
        scopePenalty: 1,
        clarityScore: 0.9,
        tierMultiplier: 1,
        wasRecallBoosted: false,
        timeDecay: 0.9,
        salienceScore: 1.1,
        salienceDecay: 0.9,
        salienceReinforcement: 1.2,
        reinforcementFactor: 1.2,
        referenceBoost: 1.2,
        references: ["mem_root"],
        ageDays: 1,
      },
    ];
    searchImpl = async () => [];

    const cfg = { agents: { list: [{ id: "main", default: true }] } };
    const tool = createMemorySearchTool({ config: cfg });
    if (!tool) {
      throw new Error("tool missing");
    }

    const result = await tool.execute("call_search_refs", { query: "linked memory content" });
    const details = result.details as {
      results: Array<{
        snippet: string;
        salienceScore?: number;
        salienceDecay?: number;
        salienceReinforcement?: number;
        referenceBoost?: number;
        references?: string[];
      }>;
    };
    const first = details.results[0];
    expect(first?.snippet).toContain("[ref:mem_root]");
    expect(first?.salienceScore).toBeCloseTo(1.1);
    expect(first?.salienceDecay).toBeCloseTo(0.9);
    expect(first?.salienceReinforcement).toBeCloseTo(1.2);
    expect(first?.referenceBoost).toBeCloseTo(1.2);
    expect(first?.references).toEqual(["mem_root"]);
  });
});

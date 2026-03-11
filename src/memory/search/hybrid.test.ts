import { afterEach, describe, expect, it, vi } from "vitest";
import { bm25RankToScore, buildFtsQuery, mergeHybridResults } from "./hybrid.js";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("memory hybrid helpers", () => {
  it("buildFtsQuery tokenizes and AND-joins", () => {
    expect(buildFtsQuery("hello world")).toBe('"hello" AND "world"');
    expect(buildFtsQuery("FOO_bar baz-1")).toBe('"FOO_bar" AND "baz" AND "1"');
    expect(buildFtsQuery("金银价格")).toBe('"金银价格"');
    expect(buildFtsQuery("価格 2026年")).toBe('"価格" AND "2026年"');
    expect(buildFtsQuery("   ")).toBeNull();
  });

  it("bm25RankToScore is monotonic and clamped", () => {
    expect(bm25RankToScore(0)).toBeCloseTo(1);
    expect(bm25RankToScore(1)).toBeCloseTo(0.5);
    expect(bm25RankToScore(10)).toBeLessThan(bm25RankToScore(1));
    expect(bm25RankToScore(-100)).toBeCloseTo(1);
  });

  it("mergeHybridResults unions by id and combines weighted scores", async () => {
    const merged = await mergeHybridResults({
      vectorWeight: 0.7,
      textWeight: 0.3,
      vector: [
        {
          id: "a",
          path: "memory/a.md",
          startLine: 1,
          endLine: 2,
          source: "memory",
          snippet: "vec-a",
          vectorScore: 0.9,
        },
      ],
      keyword: [
        {
          id: "b",
          path: "memory/b.md",
          startLine: 3,
          endLine: 4,
          source: "memory",
          snippet: "kw-b",
          textScore: 1.0,
        },
      ],
    });

    expect(merged).toHaveLength(2);
    const a = merged.find((r) => r.path === "memory/a.md");
    const b = merged.find((r) => r.path === "memory/b.md");
    expect(a?.score).toBeCloseTo(0.7 * 0.9);
    expect(b?.score).toBeCloseTo(0.3 * 1.0);
  });

  it("mergeHybridResults prefers keyword snippet when ids overlap", async () => {
    const merged = await mergeHybridResults({
      vectorWeight: 0.5,
      textWeight: 0.5,
      vector: [
        {
          id: "a",
          path: "memory/a.md",
          startLine: 1,
          endLine: 2,
          source: "memory",
          snippet: "vec-a",
          vectorScore: 0.2,
        },
      ],
      keyword: [
        {
          id: "a",
          path: "memory/a.md",
          startLine: 1,
          endLine: 2,
          source: "memory",
          snippet: "kw-a",
          textScore: 1.0,
        },
      ],
    });

    expect(merged).toHaveLength(1);
    expect(merged[0]?.snippet).toBe("kw-a");
    expect(merged[0]?.score).toBeCloseTo(0.5 * 0.2 + 0.5 * 1.0);
  });

  it("reranks results without overwriting the merged score", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 200,
        json: async () => ({
          results: [
            { index: 1, relevance_score: 0.9 },
            { index: 0, relevance_score: 0.1 },
          ],
        }),
      })),
    );

    const merged = await mergeHybridResults({
      query: "alpha",
      vectorWeight: 1,
      textWeight: 0,
      vector: [
        {
          id: "a",
          path: "memory/a.md",
          startLine: 1,
          endLine: 2,
          source: "memory",
          snippet: "alpha first",
          vectorScore: 0.9,
        },
        {
          id: "b",
          path: "memory/b.md",
          startLine: 3,
          endLine: 4,
          source: "memory",
          snippet: "alpha second",
          vectorScore: 0.8,
        },
      ],
      keyword: [],
      reranker: {
        enabled: true,
        apiUrl: "http://localhost:8081/v1/rerank",
        model: "Qwen3-Reranker-0.6B",
        maxCandidates: 2,
      },
    });

    expect(merged.map((entry) => entry.path)).toEqual(["memory/b.md", "memory/a.md"]);
    expect(merged[0]?.score).toBeCloseTo(0.8);
    expect(merged[1]?.score).toBeCloseTo(0.9);
  });

  it("falls back to hybrid ordering when reranking fails", async () => {
    const warn = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 500,
        text: async () => "boom",
      })),
    );

    const merged = await mergeHybridResults({
      query: "alpha",
      vectorWeight: 1,
      textWeight: 0,
      vector: [
        {
          id: "a",
          path: "memory/a.md",
          startLine: 1,
          endLine: 2,
          source: "memory",
          snippet: "alpha first",
          vectorScore: 0.9,
        },
        {
          id: "b",
          path: "memory/b.md",
          startLine: 3,
          endLine: 4,
          source: "memory",
          snippet: "alpha second",
          vectorScore: 0.8,
        },
      ],
      keyword: [],
      reranker: {
        enabled: true,
        apiUrl: "http://localhost:8081/v1/rerank",
        model: "Qwen3-Reranker-0.6B",
        maxCandidates: 2,
      },
      warn,
    });

    expect(merged.map((entry) => entry.path)).toEqual(["memory/a.md", "memory/b.md"]);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]?.[0]).toContain("memory reranker failed");
  });
});

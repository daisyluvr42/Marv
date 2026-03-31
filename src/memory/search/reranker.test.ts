import { beforeEach, describe, expect, it, vi } from "vitest";

const fetchWithPrivateNetworkAccess = vi.hoisted(() => vi.fn());

vi.mock("../../infra/net/private-network-fetch.js", () => ({
  fetchWithPrivateNetworkAccess,
}));

import { rerankHybridResults } from "./reranker.js";

describe("rerankHybridResults", () => {
  beforeEach(() => {
    fetchWithPrivateNetworkAccess.mockReset();
  });

  it("uses guarded private-network fetch and applies reranked order", async () => {
    const release = vi.fn(async () => {});
    fetchWithPrivateNetworkAccess.mockResolvedValueOnce({
      response: new Response(
        JSON.stringify({
          results: [
            { index: 1, relevance_score: 0.9 },
            { index: 0, relevance_score: 0.2 },
          ],
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ),
      finalUrl: "http://127.0.0.1:8080/rerank",
      release,
    });

    const results = await rerankHybridResults({
      query: "hello",
      results: [
        { id: "a", score: 0.6, snippet: "first" },
        { id: "b", score: 0.5, snippet: "second" },
      ],
      reranker: {
        enabled: true,
        apiUrl: "http://127.0.0.1:8080/rerank",
        model: "bge-reranker",
        apiKey: "test-key",
        maxCandidates: 2,
      },
    });

    expect(results.map((entry) => entry.id)).toEqual(["b", "a"]);
    expect(fetchWithPrivateNetworkAccess).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "http://127.0.0.1:8080/rerank",
        timeoutMs: 15000,
        auditContext: "memory.reranker",
      }),
    );
    expect(release).toHaveBeenCalledTimes(1);
  });

  it("uses explicit reranker headers instead of forcing bearer auth", async () => {
    const release = vi.fn(async () => {});
    fetchWithPrivateNetworkAccess.mockResolvedValueOnce({
      response: new Response(
        JSON.stringify({
          results: [{ index: 0, relevance_score: 1 }],
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ),
      finalUrl: "http://127.0.0.1:8080/rerank",
      release,
    });

    await rerankHybridResults({
      query: "hello",
      results: [
        { id: "a", score: 0.6, snippet: "first" },
        { id: "b", score: 0.5, snippet: "second" },
      ],
      reranker: {
        enabled: true,
        apiUrl: "http://127.0.0.1:8080/rerank",
        model: "bge-reranker",
        apiKey: "ignored-bearer-key",
        headers: { "X-API-Key": "custom-key" },
        maxCandidates: 2,
      },
    });

    expect(fetchWithPrivateNetworkAccess).toHaveBeenCalledWith(
      expect.objectContaining({
        init: expect.objectContaining({
          headers: expect.objectContaining({
            "Content-Type": "application/json",
            "X-API-Key": "custom-key",
          }),
        }),
      }),
    );
    const firstCall = fetchWithPrivateNetworkAccess.mock.calls[0];
    expect(firstCall).toBeDefined();
    const params = (firstCall?.[0] ?? {}) as {
      init?: { headers?: Record<string, string> };
    };
    expect(params.init?.headers?.Authorization).toBeUndefined();
  });
});

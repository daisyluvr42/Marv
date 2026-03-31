import { beforeEach, describe, expect, it, vi } from "vitest";

const fetchWithPrivateNetworkAccess = vi.hoisted(() => vi.fn());

vi.mock("../../infra/net/private-network-fetch.js", () => ({
  fetchWithPrivateNetworkAccess,
}));

import { fetchRemoteEmbeddingVectors } from "./embeddings-remote-fetch.js";

describe("fetchRemoteEmbeddingVectors", () => {
  beforeEach(() => {
    fetchWithPrivateNetworkAccess.mockReset();
  });

  it("uses guarded private-network fetch and releases the response", async () => {
    const release = vi.fn(async () => {});
    fetchWithPrivateNetworkAccess.mockResolvedValueOnce({
      response: new Response(
        JSON.stringify({
          data: [{ embedding: [0.1, 0.2] }],
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ),
      finalUrl: "http://127.0.0.1:8000/v1/embeddings",
      release,
    });

    const result = await fetchRemoteEmbeddingVectors({
      url: "http://127.0.0.1:8000/v1/embeddings",
      headers: {
        Authorization: "Bearer test",
      },
      body: {
        model: "text-embedding-3-small",
        input: ["hello"],
      },
      errorPrefix: "openai embeddings failed",
    });

    expect(result).toEqual([[0.1, 0.2]]);
    expect(fetchWithPrivateNetworkAccess).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "http://127.0.0.1:8000/v1/embeddings",
        auditContext: "memory.embeddings.remote",
      }),
    );
    expect(release).toHaveBeenCalledTimes(1);
  });
});

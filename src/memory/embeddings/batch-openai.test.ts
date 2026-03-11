import { afterEach, describe, expect, it, vi } from "vitest";
import { runOpenAiEmbeddingBatches, type OpenAiEmbeddingClient } from "./batch-openai.js";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("openai embedding batches", () => {
  it("includes dimensions in uploaded batch requests", async () => {
    const fetchMock = vi.fn(async (input?: unknown, init?: unknown) => {
      const url = String(input);
      if (url.endsWith("/files") && (init as RequestInit | undefined)?.method === "POST") {
        return {
          ok: true,
          status: 200,
          json: async () => ({ id: "file-input" }),
        };
      }
      if (url.endsWith("/batches")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            id: "batch-1",
            status: "completed",
            output_file_id: "file-output",
          }),
        };
      }
      if (url.endsWith("/files/file-output/content")) {
        return {
          ok: true,
          status: 200,
          text: async () =>
            JSON.stringify({
              custom_id: "chunk-1",
              response: {
                status_code: 200,
                body: { data: [{ index: 0, embedding: [1, 0, 0] }] },
              },
            }),
        };
      }
      throw new Error(`Unexpected fetch call: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client: OpenAiEmbeddingClient = {
      baseUrl: "http://localhost:8080/v1",
      headers: { Authorization: "Bearer local" },
      model: "Qwen3-Embedding-0.6B",
      dimensions: 512,
    };

    const result = await runOpenAiEmbeddingBatches({
      openAi: client,
      agentId: "main",
      requests: [
        {
          custom_id: "chunk-1",
          method: "POST",
          url: "/v1/embeddings",
          body: {
            model: client.model,
            input: "hello world",
            dimensions: client.dimensions,
          },
        },
      ],
      wait: true,
      pollIntervalMs: 10,
      timeoutMs: 1000,
      concurrency: 1,
    });

    expect(result.get("chunk-1")).toEqual([1, 0, 0]);

    const uploadCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/files")) as [
      unknown,
      RequestInit,
    ];
    const form = uploadCall?.[1]?.body as FormData;
    const uploaded = form.get("file");
    expect(uploaded).toBeInstanceOf(Blob);
    const jsonl = await (uploaded as Blob).text();
    const line = JSON.parse(jsonl.trim()) as {
      body?: { dimensions?: number };
    };
    expect(line.body?.dimensions).toBe(512);
  });
});

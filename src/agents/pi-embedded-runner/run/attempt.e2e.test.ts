import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { ImageContent } from "@mariozechner/pi-ai";
import { beforeEach, describe, expect, it, vi } from "vitest";

const fetchWithPrivateNetworkAccessMock = vi.fn();

vi.mock("../../../infra/net/private-network-fetch.js", () => ({
  fetchWithPrivateNetworkAccess: (params: unknown) => fetchWithPrivateNetworkAccessMock(params),
}));

import { injectHistoryImagesIntoMessages, warmOllamaModel } from "./attempt.js";

describe("injectHistoryImagesIntoMessages", () => {
  const image: ImageContent = { type: "image", data: "abc", mimeType: "image/png" };

  it("injects history images and converts string content", () => {
    const messages: AgentMessage[] = [
      {
        role: "user",
        content: "See /tmp/photo.png",
      } as AgentMessage,
    ];

    const didMutate = injectHistoryImagesIntoMessages(messages, new Map([[0, [image]]]));

    expect(didMutate).toBe(true);
    const firstUser = messages[0] as Extract<AgentMessage, { role: "user" }> | undefined;
    expect(Array.isArray(firstUser?.content)).toBe(true);
    const content = firstUser?.content as Array<{ type: string; text?: string; data?: string }>;
    expect(content).toHaveLength(2);
    expect(content[0]?.type).toBe("text");
    expect(content[1]).toMatchObject({ type: "image", data: "abc" });
  });

  it("avoids duplicating existing image content", () => {
    const messages: AgentMessage[] = [
      {
        role: "user",
        content: [{ type: "text", text: "See /tmp/photo.png" }, { ...image }],
      } as AgentMessage,
    ];

    const didMutate = injectHistoryImagesIntoMessages(messages, new Map([[0, [image]]]));

    expect(didMutate).toBe(false);
    const first = messages[0] as Extract<AgentMessage, { role: "user" }> | undefined;
    if (!first || !Array.isArray(first.content)) {
      throw new Error("expected array content");
    }
    expect(first.content).toHaveLength(2);
  });

  it("ignores non-user messages and out-of-range indices", () => {
    const messages: AgentMessage[] = [
      {
        role: "assistant",
        content: "noop",
      } as unknown as AgentMessage,
    ];

    const didMutate = injectHistoryImagesIntoMessages(messages, new Map([[1, [image]]]));

    expect(didMutate).toBe(false);
    const firstAssistant = messages[0] as Extract<AgentMessage, { role: "assistant" }> | undefined;
    expect(firstAssistant?.content).toBe("noop");
  });
});

describe("warmOllamaModel", () => {
  beforeEach(() => {
    fetchWithPrivateNetworkAccessMock.mockReset();
    fetchWithPrivateNetworkAccessMock.mockResolvedValue({
      response: new Response("{}", { status: 200 }),
      release: vi.fn(async () => {}),
    });
  });

  it("uses the resolved runtime model id and normalized configured base URL", async () => {
    await warmOllamaModel({
      baseUrl: "http://192.168.0.42:11434/v1",
      modelId: "qwen3.5:122b-a10b",
    });

    expect(fetchWithPrivateNetworkAccessMock).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "http://192.168.0.42:11434/api/generate",
        auditContext: "agent.ollama.warmup",
      }),
    );

    const call = fetchWithPrivateNetworkAccessMock.mock.calls[0]?.[0] as {
      init?: { body?: string };
    };
    expect(call.init?.body).toBeDefined();
    expect(JSON.parse(call.init?.body ?? "{}")).toEqual({
      model: "qwen3.5:122b-a10b",
      prompt: "",
      stream: false,
      keep_alive: "60m",
    });
  });
});

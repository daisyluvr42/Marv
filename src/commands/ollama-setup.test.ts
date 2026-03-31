import { beforeEach, describe, expect, it, vi } from "vitest";

const fetchWithPrivateNetworkAccess = vi.hoisted(() => vi.fn());

vi.mock("../infra/net/private-network-fetch.js", () => ({
  fetchWithPrivateNetworkAccess,
}));

import { promptAndConfigureOllama } from "./ollama-setup.js";

describe("promptAndConfigureOllama", () => {
  beforeEach(() => {
    fetchWithPrivateNetworkAccess.mockReset();
  });

  it("verifies the configured model against the Ollama tags endpoint", async () => {
    const release = vi.fn(async () => {});
    fetchWithPrivateNetworkAccess.mockResolvedValueOnce({
      response: new Response(
        JSON.stringify({
          models: [{ name: "qwen2.5-coder:latest" }],
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ),
      finalUrl: "http://127.0.0.1:11434/api/tags",
      release,
    });

    const progressStop = vi.fn();
    const result = await promptAndConfigureOllama({
      cfg: {},
      prompter: {
        text: vi
          .fn()
          .mockResolvedValueOnce("http://127.0.0.1:11434")
          .mockResolvedValueOnce("")
          .mockResolvedValueOnce("qwen2.5-coder:latest"),
        progress: vi.fn(() => ({ update: vi.fn(), stop: progressStop })),
      } as never,
    });

    expect(fetchWithPrivateNetworkAccess).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "http://127.0.0.1:11434/api/tags",
        auditContext: "onboard.ollama.verify",
      }),
    );
    expect(result.modelRef).toBe("ollama/qwen2.5-coder:latest");
    expect(result.config.models?.providers?.ollama?.api).toBe("ollama");
    expect(result.config.models?.providers?.ollama?.models?.[0]?.id).toBe("qwen2.5-coder:latest");
    expect(progressStop).toHaveBeenCalledWith("Ollama endpoint verified.");
    expect(release).toHaveBeenCalledTimes(1);
  });
});

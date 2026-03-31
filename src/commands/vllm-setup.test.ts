import { beforeEach, describe, expect, it, vi } from "vitest";

const upsertAuthProfileWithLock = vi.hoisted(() => vi.fn(async () => {}));
const fetchWithPrivateNetworkAccess = vi.hoisted(() => vi.fn());

vi.mock("../agents/auth-profiles.js", () => ({
  upsertAuthProfileWithLock,
}));

vi.mock("../infra/net/private-network-fetch.js", () => ({
  fetchWithPrivateNetworkAccess,
}));

import { promptAndConfigureVllm } from "./vllm-setup.js";

describe("promptAndConfigureVllm", () => {
  beforeEach(() => {
    upsertAuthProfileWithLock.mockReset();
    fetchWithPrivateNetworkAccess.mockReset();
  });

  it("verifies the configured model before writing auth/config", async () => {
    const release = vi.fn(async () => {});
    fetchWithPrivateNetworkAccess.mockResolvedValueOnce({
      response: new Response(
        JSON.stringify({
          data: [{ id: "meta-llama/Meta-Llama-3-8B-Instruct" }],
        }),
        {
          status: 200,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ),
      finalUrl: "http://127.0.0.1:8000/v1/models",
      release,
    });

    const progressStop = vi.fn();
    const result = await promptAndConfigureVllm({
      cfg: {},
      prompter: {
        text: vi
          .fn()
          .mockResolvedValueOnce("http://127.0.0.1:8000/v1")
          .mockResolvedValueOnce("sk-vllm")
          .mockResolvedValueOnce("meta-llama/Meta-Llama-3-8B-Instruct"),
        progress: vi.fn(() => ({ update: vi.fn(), stop: progressStop })),
      } as never,
      agentDir: "/tmp/agent",
    });

    expect(fetchWithPrivateNetworkAccess).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "http://127.0.0.1:8000/v1/models",
        auditContext: "onboard.vllm.verify",
      }),
    );
    expect(upsertAuthProfileWithLock).toHaveBeenCalledWith(
      expect.objectContaining({
        profileId: "vllm:default",
        agentDir: "/tmp/agent",
      }),
    );
    expect(result.modelRef).toBe("vllm/meta-llama/Meta-Llama-3-8B-Instruct");
    expect(result.config.models?.providers?.vllm?.models?.[0]?.id).toBe(
      "meta-llama/Meta-Llama-3-8B-Instruct",
    );
    expect(progressStop).toHaveBeenCalledWith("vLLM endpoint verified.");
    expect(release).toHaveBeenCalledTimes(1);
  });
});

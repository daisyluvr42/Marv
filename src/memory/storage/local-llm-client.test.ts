import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";

const fetchWithSsrFGuardMock = vi.fn();

vi.mock("../../infra/net/fetch-guard.js", () => ({
  fetchWithSsrFGuard: (...args: unknown[]) => fetchWithSsrFGuardMock(...args),
}));

import { inferLocal, probeLocalModel } from "./local-llm-client.js";

describe("local-llm-client", () => {
  const previousLocalKey = process.env.LOCAL_LLM_KEY;

  beforeEach(() => {
    fetchWithSsrFGuardMock.mockReset();
    process.env.LOCAL_LLM_KEY = "secret-token";
  });

  afterEach(() => {
    if (previousLocalKey === undefined) {
      delete process.env.LOCAL_LLM_KEY;
    } else {
      process.env.LOCAL_LLM_KEY = previousLocalKey;
    }
  });

  it("probes ollama using the native api base when provider baseUrl includes /v1", async () => {
    fetchWithSsrFGuardMock.mockResolvedValue({
      response: new Response(JSON.stringify({ models: [] }), { status: 200 }),
      release: async () => {},
    });

    const result = await probeLocalModel({
      cfg: {
        models: {
          providers: {
            ollama: {
              baseUrl: "http://ollama-host:11434/v1",
              api: "ollama",
              models: [],
            },
          },
        },
      } as MarvConfig,
      model: {
        provider: "ollama",
        model: "test-model",
      },
    });

    expect(result.ok).toBe(true);
    expect(fetchWithSsrFGuardMock).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "http://ollama-host:11434/api/tags",
      }),
    );
  });

  it("calls openai-compatible chat completions with inherited provider auth", async () => {
    fetchWithSsrFGuardMock.mockResolvedValue({
      response: new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content: "semantic summary",
              },
            },
          ],
        }),
        { status: 200 },
      ),
      release: async () => {},
    });

    const result = await inferLocal({
      cfg: {
        models: {
          providers: {
            vllm: {
              baseUrl: "http://127.0.0.1:8000/v1",
              api: "openai-completions",
              apiKey: "LOCAL_LLM_KEY",
              headers: {
                "X-Test": "1",
              },
              models: [
                {
                  id: "local-model",
                  name: "Local Model",
                  reasoning: false,
                  input: ["text"],
                  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                  contextWindow: 32000,
                  maxTokens: 4096,
                },
              ],
            },
          },
        },
      } as MarvConfig,
      model: {
        provider: "vllm",
      },
      system: "system prompt",
      prompt: "user prompt",
    });

    expect(result).toEqual({ ok: true, text: "semantic summary" });
    expect(fetchWithSsrFGuardMock).toHaveBeenCalledWith(
      expect.objectContaining({
        url: "http://127.0.0.1:8000/v1/chat/completions",
        init: expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            Authorization: "Bearer secret-token",
            "Content-Type": "application/json",
            "X-Test": "1",
          }),
        }),
      }),
    );
  });

  it("returns an error when no model is configured", async () => {
    const result = await inferLocal({
      cfg: {} as MarvConfig,
      system: "system prompt",
      prompt: "user prompt",
    });

    expect(result.ok).toBe(false);
    expect(result).toEqual(
      expect.objectContaining({
        error: expect.stringContaining("no model configured"),
      }),
    );
    expect(fetchWithSsrFGuardMock).not.toHaveBeenCalled();
  });

  it("returns an error for unsupported provider apis", async () => {
    const result = await inferLocal({
      cfg: {
        models: {
          providers: {
            anthropic: {
              baseUrl: "https://api.anthropic.com",
              api: "anthropic-messages",
              models: [],
            },
          },
        },
      } as MarvConfig,
      model: {
        provider: "anthropic",
      },
      system: "system prompt",
      prompt: "user prompt",
    });

    expect(result.ok).toBe(false);
    expect(result).toEqual(
      expect.objectContaining({
        error: expect.stringContaining("deep-consolidation model api"),
      }),
    );
    expect(fetchWithSsrFGuardMock).not.toHaveBeenCalled();
  });
});

import crypto from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";
import type { AuthProfileStore } from "../auth-profiles.js";
import { saveAuthProfileStore } from "../auth-profiles.js";
import { AUTH_STORE_VERSION } from "../auth-profiles/constants.js";
import { runWithModelFallback } from "./model-fallback.js";

function makeCfg(overrides: Partial<MarvConfig> = {}): MarvConfig {
  return {
    agents: {
      defaults: {
        model: {
          primary: "openai/gpt-4.1-mini",
          fallbacks: ["anthropic/claude-haiku-3-5"],
        },
      },
    },
    ...overrides,
  } as MarvConfig;
}

function makeFallbacksOnlyCfg(): MarvConfig {
  return {
    agents: {
      defaults: {
        model: {
          fallbacks: ["openai/gpt-5.2"],
        },
      },
    },
  } as MarvConfig;
}

function makeProviderFallbackCfg(provider: string): MarvConfig {
  return makeCfg({
    agents: {
      defaults: {
        model: {
          primary: `${provider}/m1`,
          fallbacks: ["fallback/ok-model"],
        },
      },
    },
  });
}

async function withTempAuthStore<T>(
  store: AuthProfileStore,
  run: (tempDir: string) => Promise<T>,
): Promise<T> {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-auth-"));
  saveAuthProfileStore(store, tempDir);
  try {
    return await run(tempDir);
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

async function runWithStoredAuth(params: {
  cfg: MarvConfig;
  store: AuthProfileStore;
  provider: string;
  run: (provider: string, model: string) => Promise<string>;
}) {
  return withTempAuthStore(params.store, async (tempDir) =>
    runWithModelFallback({
      cfg: params.cfg,
      provider: params.provider,
      model: "m1",
      agentDir: tempDir,
      run: params.run,
    }),
  );
}

async function expectFallsBackToHaiku(params: {
  provider: string;
  model: string;
  firstError: Error;
}) {
  const cfg = makeCfg();
  const run = vi.fn().mockRejectedValueOnce(params.firstError).mockResolvedValueOnce("ok");

  const result = await runWithModelFallback({
    cfg,
    provider: params.provider,
    model: params.model,
    run,
  });

  expect(result.result).toBe("ok");
  expect(run).toHaveBeenCalledTimes(2);
  expect(run.mock.calls[1]?.[0]).toBe("anthropic");
  expect(run.mock.calls[1]?.[1]).toBe("claude-haiku-3-5");
}

describe("runWithModelFallback", () => {
  it("preserves explicit openai gpt-5.3 codex provider before running", async () => {
    const cfg = makeCfg();
    const run = vi.fn().mockResolvedValueOnce("ok");

    const result = await runWithModelFallback({
      cfg,
      provider: "openai",
      model: "gpt-5.3-codex",
      run,
    });

    expect(result.result).toBe("ok");
    expect(run).toHaveBeenCalledTimes(1);
    expect(run).toHaveBeenCalledWith("openai", "gpt-5.3-codex");
  });

  it("does not fall back on non-auth errors", async () => {
    const cfg = makeCfg();
    const run = vi.fn().mockRejectedValueOnce(new Error("bad request")).mockResolvedValueOnce("ok");

    await expect(
      runWithModelFallback({
        cfg,
        provider: "openai",
        model: "gpt-4.1-mini",
        run,
      }),
    ).rejects.toThrow("bad request");
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("falls back on auth errors", async () => {
    await expectFallsBackToHaiku({
      provider: "openai",
      model: "gpt-4.1-mini",
      firstError: Object.assign(new Error("nope"), { status: 401 }),
    });
  });

  it("falls back on transient HTTP 5xx errors", async () => {
    await expectFallsBackToHaiku({
      provider: "openai",
      model: "gpt-4.1-mini",
      firstError: new Error(
        "521 <!DOCTYPE html><html><head><title>Web server is down</title></head><body>Cloudflare</body></html>",
      ),
    });
  });

  it("falls back on 402 payment required", async () => {
    await expectFallsBackToHaiku({
      provider: "openai",
      model: "gpt-4.1-mini",
      firstError: Object.assign(new Error("payment required"), { status: 402 }),
    });
  });

  it("falls back on billing errors", async () => {
    await expectFallsBackToHaiku({
      provider: "openai",
      model: "gpt-4.1-mini",
      firstError: new Error(
        "LLM request rejected: Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.",
      ),
    });
  });

  it("routes assigned models through the configured primary before configured fallbacks", async () => {
    const cfg = makeCfg();
    const calls: Array<{ provider: string; model: string }> = [];

    const result = await runWithModelFallback({
      cfg,
      provider: "anthropic",
      model: "claude-opus-4",
      run: async (provider, model) => {
        calls.push({ provider, model });
        if (calls.length === 1) {
          throw new Error('No credentials found for profile "anthropic:default".');
        }
        return "ok";
      },
    });

    expect(result.result).toBe("ok");
    expect(calls).toEqual([
      { provider: "anthropic", model: "claude-opus-4" },
      { provider: "openai", model: "gpt-4.1-mini" },
    ]);
  });

  it("continues through the configured route until a model succeeds", async () => {
    const cfg = makeCfg();
    const calls: Array<{ provider: string; model: string }> = [];

    const result = await runWithModelFallback({
      cfg,
      provider: "anthropic",
      model: "claude-opus-4",
      run: async (provider, model) => {
        calls.push({ provider, model });
        if (calls.length < 3) {
          throw Object.assign(new Error("nope"), { status: 401 });
        }
        return "ok";
      },
    });

    expect(result.result).toBe("ok");
    expect(calls).toEqual([
      { provider: "anthropic", model: "claude-opus-4" },
      { provider: "openai", model: "gpt-4.1-mini" },
      { provider: "anthropic", model: "claude-haiku-3-5" },
    ]);
  });

  it("falls back on local connection errors without mutating model selections", async () => {
    const cfg = makeCfg({
      agents: {
        defaults: {
          model: {
            primary: "lmstudio/qwen-local",
            fallbacks: ["openai/gpt-4.1-mini"],
          },
        },
      },
      models: {
        providers: {
          lmstudio: {
            baseUrl: "http://127.0.0.1:1234/v1",
            models: [
              {
                id: "qwen-local",
                name: "Qwen Local",
                reasoning: false,
                input: ["text"],
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                contextWindow: 131_072,
                maxTokens: 16_384,
              },
            ],
          },
        },
        selections: {
          lmstudio: ["lmstudio/qwen-local"],
          openai: ["openai/gpt-4.1-mini"],
        },
      },
    });

    const result = await runWithModelFallback({
      cfg,
      provider: "lmstudio",
      model: "qwen-local",
      run: async (provider, model) => {
        if (provider === "lmstudio") {
          throw Object.assign(new Error("Connection error."), { code: "ECONNREFUSED" });
        }
        expect(model).toBe("gpt-4.1-mini");
        return "ok";
      },
    });

    expect(result.result).toBe("ok");
    expect(result.provider).toBe("openai");
    expect(result.notice).toContain("Local model lmstudio/qwen-local is unavailable");
    expect(cfg.models?.selections?.lmstudio).toEqual(["lmstudio/qwen-local"]);
  });

  it("does not label public baseUrl provider failures as local", async () => {
    const cfg = makeCfg({
      agents: {
        defaults: {
          model: {
            primary: "custom-cloud/cloud-model",
            fallbacks: ["openai/gpt-4.1-mini"],
          },
        },
      },
      models: {
        providers: {
          "custom-cloud": {
            baseUrl: "https://api.example.com/v1",
            models: [
              {
                id: "cloud-model",
                name: "Cloud Model",
                reasoning: false,
                input: ["text"],
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                contextWindow: 131_072,
                maxTokens: 16_384,
              },
            ],
          },
        },
        selections: {
          "custom-cloud": ["custom-cloud/cloud-model"],
          openai: ["openai/gpt-4.1-mini"],
        },
      },
    });

    const result = await runWithModelFallback({
      cfg,
      provider: "custom-cloud",
      model: "cloud-model",
      run: async (provider) => {
        if (provider === "custom-cloud") {
          throw Object.assign(new Error("Connection error."), { code: "ECONNRESET" });
        }
        return "ok";
      },
    });

    expect(result.result).toBe("ok");
    expect(result.provider).toBe("openai");
    expect(result.notice).toBeUndefined();
  });

  it("skips providers when all profiles are in cooldown", async () => {
    const provider = `cooldown-test-${crypto.randomUUID()}`;
    const profileId = `${provider}:default`;

    const store: AuthProfileStore = {
      version: AUTH_STORE_VERSION,
      profiles: {
        [profileId]: {
          type: "api_key",
          provider,
          key: "test-key",
        },
      },
      usageStats: {
        [profileId]: {
          cooldownUntil: Date.now() + 5 * 60_000,
        },
      },
    };

    const cfg = makeProviderFallbackCfg(provider);
    const run = vi.fn().mockImplementation(async (providerId, modelId) => {
      if (providerId === "fallback") {
        return "ok";
      }
      throw new Error(`unexpected provider: ${providerId}/${modelId}`);
    });

    const result = await runWithStoredAuth({
      cfg,
      store,
      provider,
      run,
    });

    expect(result.result).toBe("ok");
    expect(run.mock.calls).toEqual([["fallback", "ok-model"]]);
    expect(result.attempts[0]?.reason).toBe("rate_limit");
  });

  it("does not skip when any profile is available", async () => {
    const provider = `cooldown-mixed-${crypto.randomUUID()}`;
    const profileA = `${provider}:a`;
    const profileB = `${provider}:b`;

    const store: AuthProfileStore = {
      version: AUTH_STORE_VERSION,
      profiles: {
        [profileA]: {
          type: "api_key",
          provider,
          key: "key-a",
        },
        [profileB]: {
          type: "api_key",
          provider,
          key: "key-b",
        },
      },
      usageStats: {
        [profileA]: {
          cooldownUntil: Date.now() + 60_000,
        },
      },
    };

    const cfg = makeProviderFallbackCfg(provider);
    const run = vi.fn().mockImplementation(async (providerId) => {
      if (providerId === provider) {
        return "ok";
      }
      return "unexpected";
    });

    const result = await runWithStoredAuth({
      cfg,
      store,
      provider,
      run,
    });

    expect(result.result).toBe("ok");
    expect(run.mock.calls).toEqual([[provider, "m1"]]);
    expect(result.attempts).toEqual([]);
  });

  it("does not append configured primary when fallbacksOverride is set", async () => {
    const cfg = makeCfg({
      agents: {
        defaults: {
          model: {
            primary: "openai/gpt-4.1-mini",
          },
        },
      },
    });
    const run = vi
      .fn()
      .mockImplementation(() => Promise.reject(Object.assign(new Error("nope"), { status: 401 })));

    await expect(
      runWithModelFallback({
        cfg,
        provider: "anthropic",
        model: "claude-opus-4-5",
        fallbacksOverride: ["anthropic/claude-haiku-3-5"],
        run,
      }),
    ).rejects.toThrow("All models failed");

    expect(run.mock.calls).toEqual([
      ["anthropic", "claude-opus-4-5"],
      ["anthropic", "claude-haiku-3-5"],
    ]);
  });

  it("uses explicit fallback candidates instead of planner defaults", async () => {
    const cfg = makeFallbacksOnlyCfg();

    const calls: Array<{ provider: string; model: string }> = [];

    const res = await runWithModelFallback({
      cfg,
      provider: "anthropic",
      model: "claude-opus-4-5",
      fallbacksOverride: ["openai/gpt-4.1"],
      run: async (provider, model) => {
        calls.push({ provider, model });
        if (provider === "anthropic") {
          throw Object.assign(new Error("nope"), { status: 401 });
        }
        if (provider === "openai" && model === "gpt-4.1") {
          return "ok";
        }
        throw new Error(`unexpected candidate: ${provider}/${model}`);
      },
    });

    expect(res.result).toBe("ok");
    expect(calls).toEqual([
      { provider: "anthropic", model: "claude-opus-4-5" },
      { provider: "openai", model: "gpt-4.1" },
    ]);
  });

  it("treats an empty fallbacksOverride as disabling global fallbacks", async () => {
    const cfg = makeFallbacksOnlyCfg();

    const calls: Array<{ provider: string; model: string }> = [];

    await expect(
      runWithModelFallback({
        cfg,
        provider: "anthropic",
        model: "claude-opus-4-5",
        fallbacksOverride: [],
        run: async (provider, model) => {
          calls.push({ provider, model });
          throw new Error("primary failed");
        },
      }),
    ).rejects.toThrow("primary failed");

    expect(calls).toEqual([{ provider: "anthropic", model: "claude-opus-4-5" }]);
  });

  it("defaults provider/model when missing (regression #946)", async () => {
    const cfg = makeCfg({
      agents: {
        defaults: {
          model: {
            primary: "openai/gpt-4.1-mini",
            fallbacks: [],
          },
        },
      },
    });

    const calls: Array<{ provider: string; model: string }> = [];

    const result = await runWithModelFallback({
      cfg,
      provider: undefined as unknown as string,
      model: undefined as unknown as string,
      run: async (provider, model) => {
        calls.push({ provider, model });
        return "ok";
      },
    });

    expect(result.result).toBe("ok");
    expect(calls).toEqual([{ provider: "openai", model: "gpt-4.1-mini" }]);
  });

  it("falls back on missing API key errors", async () => {
    await expectFallsBackToHaiku({
      provider: "openai",
      model: "gpt-4.1-mini",
      firstError: new Error("No API key found for profile openai."),
    });
  });

  it("falls back on lowercase credential errors", async () => {
    await expectFallsBackToHaiku({
      provider: "openai",
      model: "gpt-4.1-mini",
      firstError: new Error("no api key found for profile openai"),
    });
  });

  it("falls back on timeout abort errors", async () => {
    const timeoutCause = Object.assign(new Error("request timed out"), { name: "TimeoutError" });
    await expectFallsBackToHaiku({
      provider: "openai",
      model: "gpt-4.1-mini",
      firstError: Object.assign(new Error("aborted"), { name: "AbortError", cause: timeoutCause }),
    });
  });

  it("falls back on abort errors with timeout reasons", async () => {
    await expectFallsBackToHaiku({
      provider: "openai",
      model: "gpt-4.1-mini",
      firstError: Object.assign(new Error("aborted"), {
        name: "AbortError",
        reason: "deadline exceeded",
      }),
    });
  });

  it("falls back on abort errors with reason: abort", async () => {
    await expectFallsBackToHaiku({
      provider: "openai",
      model: "gpt-4.1-mini",
      firstError: Object.assign(new Error("aborted"), {
        name: "AbortError",
        reason: "reason: abort",
      }),
    });
  });

  it("falls back when message says aborted but error is a timeout", async () => {
    await expectFallsBackToHaiku({
      provider: "openai",
      model: "gpt-4.1-mini",
      firstError: Object.assign(new Error("request aborted"), { code: "ETIMEDOUT" }),
    });
  });

  it("falls back on provider abort errors with request-aborted messages", async () => {
    await expectFallsBackToHaiku({
      provider: "openai",
      model: "gpt-4.1-mini",
      firstError: Object.assign(new Error("Request was aborted"), { name: "AbortError" }),
    });
  });

  it("does not fall back on user aborts", async () => {
    const cfg = makeCfg();
    const run = vi
      .fn()
      .mockRejectedValueOnce(Object.assign(new Error("aborted"), { name: "AbortError" }))
      .mockResolvedValueOnce("ok");

    await expect(
      runWithModelFallback({
        cfg,
        provider: "openai",
        model: "gpt-4.1-mini",
        run,
      }),
    ).rejects.toThrow("aborted");

    expect(run).toHaveBeenCalledTimes(1);
  });

  it("appends the configured primary as a last fallback", async () => {
    const cfg = makeCfg({
      agents: {
        defaults: {
          model: {
            primary: "openai/gpt-4.1-mini",
            fallbacks: [],
          },
        },
      },
    });
    const run = vi
      .fn()
      .mockRejectedValueOnce(Object.assign(new Error("timeout"), { code: "ETIMEDOUT" }))
      .mockResolvedValueOnce("ok");

    const result = await runWithModelFallback({
      cfg,
      provider: "openrouter",
      model: "meta-llama/llama-3.3-70b:free",
      run,
    });

    expect(result.result).toBe("ok");
    expect(run).toHaveBeenCalledTimes(2);
    expect(result.provider).toBe("openai");
    expect(result.model).toBe("gpt-4.1-mini");
  });
});

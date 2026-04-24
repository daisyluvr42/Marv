import "./run.overflow-compaction.mocks.shared.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { resolveModel } from "./model.js";
import { runEmbeddedPiAgent } from "./run.js";
import { runEmbeddedAttempt } from "./run/attempt.js";

const mockedResolveModel = vi.mocked(resolveModel);
const mockedRunEmbeddedAttempt = vi.mocked(runEmbeddedAttempt);

describe("runEmbeddedPiAgent provider timeout", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedResolveModel.mockReturnValue({
      model: {
        id: "qwen3.5:122b-a10b",
        provider: "local-qwen",
        contextWindow: 200000,
        api: "openai-completions",
        baseUrl: "http://10.0.0.1:11434/v1",
      },
      error: null,
      authStorage: {
        setRuntimeApiKey: vi.fn(),
      },
      modelRegistry: {},
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
    mockedRunEmbeddedAttempt.mockResolvedValue({
      aborted: false,
      promptError: null,
      timedOut: false,
      sessionIdUsed: "test-session",
      assistantTexts: ["OK"],
      lastAssistant: {
        usage: { input: 1, output: 1, total: 2 },
        stopReason: "end_turn",
      },
      attemptUsage: { input: 1, output: 1, total: 2 },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
  });

  it("uses the configured provider timeout when it exceeds the caller timeout", async () => {
    await runEmbeddedPiAgent({
      sessionId: "test-session",
      sessionKey: "test-key",
      sessionFile: "/tmp/session.json",
      workspaceDir: "/tmp/workspace",
      config: {
        models: {
          providers: {
            "local-qwen": {
              baseUrl: "http://10.0.0.1:11434/v1",
              timeoutMs: 120_000,
              models: [
                {
                  id: "qwen3.5:122b-a10b",
                  name: "Qwen 3.5 122B",
                  reasoning: true,
                  input: ["text"],
                  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                  contextWindow: 200000,
                  maxTokens: 8192,
                },
              ],
            },
          },
        },
      },
      prompt: "hello",
      provider: "local-qwen",
      model: "qwen3.5:122b-a10b",
      timeoutMs: 30_000,
      runId: "run-1",
    });

    expect(mockedRunEmbeddedAttempt).toHaveBeenCalledWith(
      expect.objectContaining({
        timeoutMs: 120_000,
      }),
    );
  });

  it("keeps the caller timeout when it already exceeds the provider timeout", async () => {
    await runEmbeddedPiAgent({
      sessionId: "test-session",
      sessionKey: "test-key",
      sessionFile: "/tmp/session.json",
      workspaceDir: "/tmp/workspace",
      config: {
        models: {
          providers: {
            "local-qwen": {
              baseUrl: "http://10.0.0.1:11434/v1",
              timeoutMs: 60_000,
              models: [
                {
                  id: "qwen3.5:122b-a10b",
                  name: "Qwen 3.5 122B",
                  reasoning: true,
                  input: ["text"],
                  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                  contextWindow: 200000,
                  maxTokens: 8192,
                },
              ],
            },
          },
        },
      },
      prompt: "hello",
      provider: "local-qwen",
      model: "qwen3.5:122b-a10b",
      timeoutMs: 180_000,
      runId: "run-2",
    });

    expect(mockedRunEmbeddedAttempt).toHaveBeenCalledWith(
      expect.objectContaining({
        timeoutMs: 180_000,
      }),
    );
  });
});

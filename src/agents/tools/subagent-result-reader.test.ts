import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../../core/gateway/call.js", () => ({
  callGateway: vi.fn(),
}));

describe("readSubagentResult", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("waits for completion and extracts assistant text", async () => {
    const { callGateway } = await import("../../core/gateway/call.js");
    const mock = vi.mocked(callGateway);
    mock.mockImplementation(async (opts: { method: string }) => {
      if (opts.method === "agent.wait") {
        return { status: "ok" };
      }
      if (opts.method === "chat.history") {
        return {
          messages: [
            { role: "user", content: [{ type: "text", text: "hello" }] },
            { role: "assistant", content: [{ type: "text", text: "result text" }] },
          ],
        };
      }
      return {};
    });

    const { readSubagentResult } = await import("./subagent-result-reader.js");
    const result = await readSubagentResult({
      runId: "run-1",
      childSessionKey: "agent:main:subagent:abc",
      waitTimeoutMs: 5000,
    });

    expect(result.status).toBe("ok");
    expect(result.text).toBe("result text");
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it("skips wait when alreadyEnded is true", async () => {
    const { callGateway } = await import("../../core/gateway/call.js");
    const mock = vi.mocked(callGateway);
    mock.mockImplementation(async (opts: { method: string }) => {
      if (opts.method === "agent.wait") {
        throw new Error("should not be called");
      }
      if (opts.method === "chat.history") {
        return {
          messages: [{ role: "assistant", content: [{ type: "text", text: "done" }] }],
        };
      }
      return {};
    });

    const { readSubagentResult } = await import("./subagent-result-reader.js");
    const result = await readSubagentResult({
      runId: "run-2",
      childSessionKey: "agent:main:subagent:def",
      waitTimeoutMs: 5000,
      alreadyEnded: true,
    });

    expect(result.status).toBe("ok");
    expect(result.text).toBe("done");
  });

  it("returns timeout status when wait reports timeout", async () => {
    const { callGateway } = await import("../../core/gateway/call.js");
    const mock = vi.mocked(callGateway);
    mock.mockImplementation(async (opts: { method: string }) => {
      if (opts.method === "agent.wait") {
        return { status: "timeout", error: "exceeded deadline" };
      }
      return {};
    });

    const { readSubagentResult } = await import("./subagent-result-reader.js");
    const result = await readSubagentResult({
      runId: "run-3",
      childSessionKey: "agent:main:subagent:ghi",
      waitTimeoutMs: 1000,
    });

    expect(result.status).toBe("timeout");
    expect(result.text).toContain("exceeded deadline");
  });

  it("returns error status when wait throws", async () => {
    const { callGateway } = await import("../../core/gateway/call.js");
    const mock = vi.mocked(callGateway);
    mock.mockImplementation(async () => {
      throw new Error("connection lost");
    });

    const { readSubagentResult } = await import("./subagent-result-reader.js");
    const result = await readSubagentResult({
      runId: "run-4",
      childSessionKey: "agent:main:subagent:jkl",
      waitTimeoutMs: 1000,
    });

    expect(result.status).toBe("error");
    expect(result.text).toContain("connection lost");
  });

  it("returns fallback text when no assistant message found", async () => {
    const { callGateway } = await import("../../core/gateway/call.js");
    const mock = vi.mocked(callGateway);
    mock.mockImplementation(async (opts: { method: string }) => {
      if (opts.method === "agent.wait") {
        return { status: "ok" };
      }
      if (opts.method === "chat.history") {
        return { messages: [{ role: "user", content: [{ type: "text", text: "task" }] }] };
      }
      return {};
    });

    const { readSubagentResult } = await import("./subagent-result-reader.js");
    const result = await readSubagentResult({
      runId: "run-5",
      childSessionKey: "agent:main:subagent:mno",
      waitTimeoutMs: 5000,
    });

    expect(result.status).toBe("ok");
    expect(result.text).toBe("(no assistant output)");
  });
});

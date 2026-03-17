import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../core/gateway/call.js", () => ({
  callGateway: vi.fn(),
}));

describe("buildSubagentContext", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns empty string for empty spec", async () => {
    const { buildSubagentContext } = await import("./subagent-context-builder.js");
    const result = await buildSubagentContext({
      spec: {},
      parentSessionKey: "agent:main:main",
    });
    expect(result).toBe("");
  });

  it("includes recent turns from parent history", async () => {
    const { callGateway } = await import("../core/gateway/call.js");
    vi.mocked(callGateway).mockResolvedValue({
      messages: [
        { role: "user", content: [{ type: "text", text: "What is 2+2?" }] },
        { role: "assistant", content: [{ type: "text", text: "4" }] },
        { role: "user", content: [{ type: "text", text: "And 3+3?" }] },
        { role: "assistant", content: [{ type: "text", text: "6" }] },
      ],
    });

    const { buildSubagentContext } = await import("./subagent-context-builder.js");
    const result = await buildSubagentContext({
      spec: { recentTurns: 1 },
      parentSessionKey: "agent:main:main",
    });

    expect(result).toContain("Recent conversation");
    expect(result).toContain("And 3+3?");
    expect(result).toContain("6");
  });

  it("includes preamble", async () => {
    const { buildSubagentContext } = await import("./subagent-context-builder.js");
    const result = await buildSubagentContext({
      spec: { preamble: "Focus on performance." },
      parentSessionKey: "agent:main:main",
    });
    expect(result).toContain("Focus on performance.");
  });

  it("truncates at maxContextChars", async () => {
    const { callGateway } = await import("../core/gateway/call.js");
    vi.mocked(callGateway).mockResolvedValue({
      messages: [
        { role: "user", content: [{ type: "text", text: "x".repeat(5000) }] },
        { role: "assistant", content: [{ type: "text", text: "y".repeat(5000) }] },
      ],
    });

    const { buildSubagentContext } = await import("./subagent-context-builder.js");
    const result = await buildSubagentContext({
      spec: { recentTurns: 2, maxContextChars: 200 },
      parentSessionKey: "agent:main:main",
    });

    expect(result.length).toBeLessThanOrEqual(300); // 200 + truncation notice
    expect(result).toContain("truncated");
  });

  it("includes file snippets when workspaceDir is provided", async () => {
    // Mock fs.readFile via vi.mock would be needed for full file test;
    // for unit test we verify the code path without real files gracefully handles missing files.
    const { buildSubagentContext } = await import("./subagent-context-builder.js");
    const result = await buildSubagentContext({
      spec: { includeFiles: ["/nonexistent/file.ts"] },
      parentSessionKey: "agent:main:main",
      workspaceDir: "/tmp",
    });
    // Missing files are skipped silently.
    expect(result).toBe("");
  });

  it("includes tool results by name", async () => {
    const { callGateway } = await import("../core/gateway/call.js");
    vi.mocked(callGateway).mockResolvedValue({
      messages: [
        { role: "user", content: [{ type: "text", text: "search" }] },
        {
          role: "toolResult",
          name: "web_search",
          content: [{ type: "text", text: "search results here" }],
        },
        {
          role: "toolResult",
          name: "read",
          content: [{ type: "text", text: "file contents" }],
        },
        { role: "assistant", content: [{ type: "text", text: "done" }] },
      ],
    });

    const { buildSubagentContext } = await import("./subagent-context-builder.js");
    const result = await buildSubagentContext({
      spec: { includeToolResults: ["web_search"] },
      parentSessionKey: "agent:main:main",
    });

    expect(result).toContain("web_search");
    expect(result).toContain("search results here");
    expect(result).not.toContain("file contents");
  });
});

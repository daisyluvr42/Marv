import { describe, expect, it, vi } from "vitest";

vi.mock("../../core/config/config.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../core/config/config.js")>();
  return {
    ...actual,
    loadConfig: () => ({
      session: { mainKey: "main", scope: "per-sender" },
      agents: {
        defaults: {
          model: { primary: "google/gemini-2.0-flash" },
          models: {},
        },
      },
    }),
  };
});

vi.mock("../../core/config/sessions.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../core/config/sessions.js")>();
  return {
    ...actual,
    resolveStorePath: () => "/tmp/main/sessions.json",
    loadSessionStore: () => ({
      "agent:main:main": {
        sessionId: "s1",
        updatedAt: 1,
        modelProvider: "google",
        model: "gemini-2.0-flash",
      },
    }),
  };
});

vi.mock("../../core/gateway/session-utils.js", () => ({
  resolveSessionModelRef: () => ({ provider: "google", model: "gemini-2.0-flash" }),
}));

vi.mock("../model/model-selection.js", () => ({
  resolveDefaultModelForAgent: () => ({ provider: "google", model: "gemini-2.5-flash" }),
}));

vi.mock("../model/model-pool.js", () => ({
  resolveRuntimeModelPlan: () => ({
    poolName: "default",
    candidates: [{ ref: "google/gemini-2.0-flash" }, { ref: "google/gemini-2.5-flash" }],
    configured: [
      { ref: "google/gemini-2.0-flash", available: true },
      { ref: "google/gemini-2.5-flash", available: true },
    ],
  }),
}));

vi.mock("../context-pollution-cleanup.js", () => ({
  inspectContextPollution: () => ({
    sessionKey: "agent:main:main",
    agentId: "main",
    preferences: {
      noPinyinChinese: true,
      noInlineEnglishChinese: false,
    },
    transcript: { violations: [], removableIds: [], sanitizedIds: [] },
    taskContext: { violations: [], removableIds: [] },
  }),
  cleanupContextPollution: async () => ({
    sessionKey: "agent:main:main",
    agentId: "main",
    preferences: {
      noPinyinChinese: true,
      noInlineEnglishChinese: false,
    },
    transcript: { violations: [], removableIds: [], sanitizedIds: [] },
    taskContext: { violations: [], removableIds: [] },
    cleaned: {
      transcriptRemoved: 2,
      taskContextRemoved: 1,
    },
  }),
  summarizeContextPollution: () => "Removable pollution detected: transcript 0, task context 0.",
}));

vi.mock("./session-status-tool.js", () => ({
  createSessionStatusTool: () => ({
    execute: async () => ({
      content: [{ type: "text", text: "runtime status" }],
      details: { statusText: "runtime status" },
    }),
  }),
}));

import { createSelfInspectingTool } from "./self-inspecting-tool.js";

describe("self_inspecting tool", () => {
  it("returns a concise summary by default", async () => {
    const tool = createSelfInspectingTool({
      agentSessionKey: "agent:main:main",
      availableToolNames: ["session_status", "self_inspecting", "self_settings"],
    });

    const result = await tool.execute("call1", {});
    const text = result.content[0]?.text ?? "";
    expect(text).toContain("Current model: google/gemini-2.0-flash");
    expect(text).toContain("Default model: google/gemini-2.5-flash");
    expect(text).toContain("Model pool: default");
  });

  it("denies cleanup for indirect instructions", async () => {
    const tool = createSelfInspectingTool({
      agentSessionKey: "agent:main:main",
      directUserInstruction: false,
      availableToolNames: ["session_status", "self_inspecting", "self_settings"],
    });

    const result = await tool.execute("call2", { cleanupContextPollution: true });
    expect(result.details).toMatchObject({ ok: false, denied: true });
  });

  it("reports cleanup results when explicitly requested", async () => {
    const tool = createSelfInspectingTool({
      agentSessionKey: "agent:main:main",
      availableToolNames: ["session_status", "self_inspecting", "self_settings"],
    });

    const result = await tool.execute("call3", { query: "context", cleanupContextPollution: true });
    const text = result.content[0]?.text ?? "";
    expect(text).toContain("Cleanup removed transcript 2, task context 1.");
    expect(result.details).toMatchObject({
      ok: true,
      cleaned: { transcriptRemoved: 2, taskContextRemoved: 1 },
    });
  });
});

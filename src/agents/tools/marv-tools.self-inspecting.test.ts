import { describe, expect, it, vi } from "vitest";

const callGatewayToolMock = vi.fn(async (method: string, ..._rest: unknown[]) => {
  if (method === "cron.status") {
    return {
      enabled: true,
      storePath: "/tmp/main/cron/jobs.json",
      nextWakeAtMs: Date.UTC(2026, 2, 9, 4, 0, 0),
    };
  }
  if (method === "cron.list") {
    return {
      jobs: [
        {
          id: "job-1",
          name: "Nightly sync",
          enabled: true,
          schedule: { kind: "cron", expr: "0 2 * * *", tz: "Asia/Shanghai" },
          state: {
            nextRunAtMs: Date.UTC(2026, 2, 9, 18, 0, 0),
            lastRunAtMs: Date.UTC(2026, 2, 8, 18, 0, 0),
            lastStatus: "ok",
          },
        },
      ],
    };
  }
  throw new Error(`unexpected gateway call: ${method}`);
});

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
        updatedAt: Date.UTC(2026, 2, 9, 3, 30, 0),
        modelProvider: "google",
        model: "gemini-2.0-flash",
        authProfileOverride: "google:work",
        thinkingLevel: "medium",
        queueMode: "collect",
        queueCap: 5,
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

vi.mock("../model/runtime-model-registry.js", () => ({
  readRuntimeModelRegistry: () => ({
    models: [
      { ref: "google/gemini-2.0-flash" },
      { ref: "google/gemini-2.5-flash" },
      { ref: "openai-codex/gpt-5.3-codex" },
    ],
    lastSuccessfulRefreshAt: Date.UTC(2026, 2, 9, 3, 0, 0),
  }),
  resolveRuntimeRegistryPathForDisplay: () => "/tmp/main/runtime/model-registry.json",
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

vi.mock("./gateway.js", () => ({
  callGatewayTool: (...args: unknown[]) => callGatewayToolMock(...(args as [string, ...unknown[]])),
}));

import { createSelfInspectingTool } from "./self-inspecting-tool.js";

describe("self_inspecting tool", () => {
  it("returns a concise summary by default", async () => {
    const tool = createSelfInspectingTool({
      agentSessionKey: "agent:main:main",
      availableToolNames: ["session_status", "self_inspecting", "self_settings"],
    });

    const result = await tool.execute("call1", {});
    const text = (result.content[0] as { text?: string } | undefined)?.text ?? "";
    expect(text).toContain("Current model: google/gemini-2.0-flash");
    expect(text).toContain("Default model: google/gemini-2.5-flash");
    expect(text).toContain("Model pool: default");
    expect(text).toContain("Scheduled jobs: 1");
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
    const text = (result.content[0] as { text?: string } | undefined)?.text ?? "";
    expect(text).toContain("Cleanup removed transcript 2, task context 1.");
    expect(result.details).toMatchObject({
      ok: true,
      cleaned: { transcriptRemoved: 2, taskContextRemoved: 1 },
    });
  });

  it("includes runtime registry details in the models report", async () => {
    const tool = createSelfInspectingTool({
      agentSessionKey: "agent:main:main",
      availableToolNames: ["session_status", "self_inspecting", "self_settings"],
    });

    const result = await tool.execute("call4", { query: "models" });
    const text = (result.content[0] as { text?: string } | undefined)?.text ?? "";
    expect(text).toContain("Runtime registry: /tmp/main/runtime/model-registry.json (3 models)");
    expect(text).toContain("Registry last refreshed: 2026-03-09T03:00:00.000Z");
    expect(text).toContain("Runnable candidate count: 2");
    expect(text).toContain("Runnable candidates: google/gemini-2.0-flash, google/gemini-2.5-flash");
    expect(result.details).toMatchObject({
      ok: true,
      models: {
        registryPath: "/tmp/main/runtime/model-registry.json",
        registryModelCount: 3,
        lastSuccessfulRefreshAt: Date.UTC(2026, 2, 9, 3, 0, 0),
      },
    });
  });

  it("maps natural-language model queries to the models section", async () => {
    const tool = createSelfInspectingTool({
      agentSessionKey: "agent:main:main",
      availableToolNames: ["cron", "self_inspecting", "self_settings"],
    });

    const result = await tool.execute("call5", { query: "available models" });
    const text = (result.content[0] as { text?: string } | undefined)?.text ?? "";
    expect(result.details).toMatchObject({ ok: true, query: "models" });
    expect(text).toContain("Runnable candidates: google/gemini-2.0-flash, google/gemini-2.5-flash");
  });

  it("reports scheduled tasks through self inspection", async () => {
    const tool = createSelfInspectingTool({
      agentSessionKey: "agent:main:main",
      availableToolNames: ["cron", "self_inspecting", "self_settings"],
    });

    const result = await tool.execute("call6", { query: "定时任务" });
    const text = (result.content[0] as { text?: string } | undefined)?.text ?? "";
    expect(result.details).toMatchObject({
      ok: true,
      query: "tasks",
      tasks: {
        enabled: true,
        storePath: "/tmp/main/cron/jobs.json",
      },
    });
    expect(text).toContain("Scheduled jobs: 1");
    expect(text).toContain("Nightly sync");
    expect(text).toContain("cron 0 2 * * * (Asia/Shanghai)");
  });

  it("includes settings, models, tasks, and tools in broad status reports", async () => {
    const tool = createSelfInspectingTool({
      agentSessionKey: "agent:main:main",
      availableToolNames: ["cron", "self_inspecting", "self_settings"],
    });

    const result = await tool.execute("call7", {
      query: "向我报告你目前的状态，包括你的定时任务，模型列表，以及其它有用的信息",
    });
    const text = (result.content[0] as { text?: string } | undefined)?.text ?? "";
    expect(result.details).toMatchObject({ ok: true, query: "all" });
    expect(text).toContain("Settings");
    expect(text).toContain("Auth profile override: google:work");
    expect(text).toContain("Models");
    expect(text).toContain("Tasks");
    expect(text).toContain("Tools");
  });
});

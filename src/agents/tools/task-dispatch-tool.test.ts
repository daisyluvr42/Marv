import { afterEach, describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";
import * as configModule from "../../core/config/config.js";
import * as gatewayModule from "../../core/gateway/call.js";
import { addSubagentRunForTests, resetSubagentRegistryForTests } from "../subagent-registry.js";
import * as spawnModule from "../subagent-spawn.js";
import { createTaskDispatchTool } from "./task-dispatch-tool.js";

describe("task_dispatch tool", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    resetSubagentRegistryForTests();
  });

  it("reuses existing role runs within the same dispatch and only spawns missing roles", async () => {
    vi.spyOn(configModule, "loadConfig").mockReturnValue({
      agents: {
        defaults: {
          subagents: {
            presets: {
              research: {
                roles: ["researcher", "fact_checker"],
              },
            },
          },
        },
      },
    } as MarvConfig);
    const spawnSpy = vi.spyOn(spawnModule, "spawnSubagentDirect").mockResolvedValue({
      status: "accepted",
      childSessionKey: "agent:main:subagent:fact-checker",
      runId: "run-fact-checker",
    });
    addSubagentRunForTests({
      runId: "run-researcher",
      childSessionKey: "agent:main:subagent:researcher",
      requesterSessionKey: "agent:main:main",
      requesterDisplayKey: "main",
      task: "Investigate launch blockers",
      cleanup: "keep",
      role: "researcher",
      preset: "research",
      taskGroup: "dispatch-1",
      dispatchId: "dispatch-1",
      announceMode: "child",
      createdAt: Date.now(),
      startedAt: Date.now(),
    });

    const tool = createTaskDispatchTool({ agentSessionKey: "agent:main:main" });
    const result = await tool.execute("call-1", {
      task: "Investigate launch blockers",
      preset: "research",
      dispatchId: "dispatch-1",
    });

    expect(result.details).toMatchObject({
      status: "accepted",
      dispatchId: "dispatch-1",
      roles: ["researcher", "fact_checker"],
      reused: [{ role: "researcher", runId: "run-researcher" }],
      spawned: [{ role: "fact_checker", runId: "run-fact-checker" }],
    });
    expect(spawnSpy).toHaveBeenCalledTimes(1);
    expect(spawnSpy.mock.calls[0]?.[0]).toMatchObject({
      role: "fact_checker",
      preset: "research",
      dispatchId: "dispatch-1",
    });
  });

  it("waits for all aggregate runs and returns collected results", async () => {
    vi.spyOn(configModule, "loadConfig").mockReturnValue({
      agents: {
        defaults: {
          subagents: {
            presets: {
              review_pack: {
                roles: ["reviewer"],
              },
            },
          },
        },
      },
    } as MarvConfig);
    vi.spyOn(spawnModule, "spawnSubagentDirect").mockImplementation(async (params) => {
      addSubagentRunForTests({
        runId: "run-reviewer",
        childSessionKey: "agent:main:subagent:reviewer",
        requesterSessionKey: "agent:main:main",
        requesterDisplayKey: "main",
        task: params.task,
        cleanup: "keep",
        role: params.role,
        preset: params.preset,
        taskGroup: params.taskGroup,
        dispatchId: params.dispatchId,
        announceMode: params.announceMode,
        createdAt: Date.now(),
        startedAt: Date.now(),
      });
      return {
        status: "accepted",
        childSessionKey: "agent:main:subagent:reviewer",
        runId: "run-reviewer",
      };
    });
    vi.spyOn(gatewayModule, "callGateway").mockImplementation(async ({ method }) => {
      if (method === "agent.wait") {
        return { status: "ok" };
      }
      if (method === "chat.history") {
        return {
          messages: [
            {
              role: "assistant",
              content: [
                { type: "text", text: "Found two regressions and one low-risk follow-up." },
              ],
            },
          ],
        };
      }
      throw new Error(`Unexpected gateway method: ${method}`);
    });

    const tool = createTaskDispatchTool({ agentSessionKey: "agent:main:main" });
    const result = await tool.execute("call-2", {
      task: "Review the patch for regressions",
      preset: "review_pack",
      waitForAll: true,
    });

    expect(result.details).toMatchObject({
      status: "ok",
      announceMode: "aggregate",
      roles: ["reviewer"],
      results: [
        {
          role: "reviewer",
          runId: "run-reviewer",
          status: "ok",
          text: "Found two regressions and one low-risk follow-up.",
        },
      ],
    });
  });
});

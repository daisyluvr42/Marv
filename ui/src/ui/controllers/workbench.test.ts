import { describe, expect, it, vi } from "vitest";
import type { WorkspaceWorkbenchState } from "./workbench.js";
import { loadWorkbench } from "./workbench.js";

describe("loadWorkbench", () => {
  it("loads the workbench snapshot through gateway RPC", async () => {
    const request = vi.fn().mockResolvedValue({
      agentId: "main",
      rows: [],
      counts: {
        active: 0,
        paused: 0,
        blocked: 0,
        queued: 0,
        completed: 0,
        archived: 0,
      },
      deliverableSummary: {
        total: 0,
        completed: 0,
      },
      fetchedAt: "2026-04-05T00:00:00.000Z",
    });
    const state: WorkspaceWorkbenchState = {
      client: {
        request,
      },
      connected: true,
      workspaceWorkbenchLoading: false,
      workspaceWorkbenchError: null,
      workspaceWorkbench: null,
    };

    await loadWorkbench(state);

    expect(request).toHaveBeenCalledWith("workbench.status", {});
    expect(state.workspaceWorkbench?.agentId).toBe("main");
    expect(state.workspaceWorkbenchLoading).toBe(false);
    expect(state.workspaceWorkbenchError).toBeNull();
  });
});

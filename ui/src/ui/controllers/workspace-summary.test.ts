import { describe, expect, it, vi } from "vitest";
import { loadWorkspaceSummary, type WorkspaceSummaryState } from "./workspace-summary.js";

type RequestFn = (method: string, params?: unknown) => Promise<unknown>;

function createState(request: RequestFn): WorkspaceSummaryState {
  return {
    client: { request } as unknown as WorkspaceSummaryState["client"],
    connected: true,
    workspaceSummaryLoading: false,
    workspaceSummaryError: null,
    workspaceSummary: null,
  };
}

describe("loadWorkspaceSummary", () => {
  it("builds the workspace summary from usage, proactive, and cron data", async () => {
    const request = vi.fn(async (method: string) => {
      if (method === "sessions.usage") {
        return {
          sessions: [{ key: "main" }, { key: "agent:main:project" }],
          totals: { totalTokens: 4200, totalCost: 1.23 },
          aggregates: {
            daily: [
              { date: "2026-03-05", messages: 0 },
              { date: "2026-03-06", messages: 4 },
              { date: "2026-03-07", messages: 9 },
            ],
          },
        };
      }
      if (method === "proactive.buffer") {
        return { pendingEntries: 3, urgentEntries: 1 };
      }
      if (method === "cron.status") {
        return { nextWakeAtMs: 1_710_000_000_000 };
      }
      if (method === "cron.list") {
        return {
          jobs: [
            { id: "job-1", state: { lastStatus: "error" } },
            { id: "job-2", state: {} },
          ],
        };
      }
      throw new Error(`unexpected method: ${method}`);
    });
    const state = createState(request);

    await loadWorkspaceSummary(state);

    expect(request).toHaveBeenCalledTimes(4);
    expect(state.workspaceSummary).toEqual(
      expect.objectContaining({
        sessionsTouched: 2,
        activeDays: 2,
        totalTokens: 4200,
        totalCost: 1.23,
        pendingProactive: 3,
        urgentProactive: 1,
        nextWakeAtMs: 1_710_000_000_000,
        failingJobs: 1,
      }),
    );
    expect(state.workspaceSummaryError).toBeNull();
    expect(state.workspaceSummaryLoading).toBe(false);
  });
});

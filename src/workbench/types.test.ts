import { describe, expect, it } from "vitest";
import {
  WORKBENCH_DEEP_LINK_SUPPORT,
  WORKBENCH_DEFAULT_ROW_LIMIT,
  WORKBENCH_POLL_INTERVAL_MS,
  WORKBENCH_PROACTIVE_GOAL_STATUS_MAP,
  WORKBENCH_PROACTIVE_TASK_STATUS_MAP,
  WORKBENCH_SUMMARY_MAX_CHARS,
  WORKBENCH_TASK_CONTEXT_STATUS_MAP,
  createEmptyWorkbenchCounts,
  truncateWorkbenchSummary,
} from "./types.js";

describe("workbench phase-0 contract", () => {
  it("locks the cross-system status mappings", () => {
    expect(WORKBENCH_TASK_CONTEXT_STATUS_MAP).toEqual({
      active: "active",
      paused: "paused",
      completed: "completed",
      archived: "archived",
    });
    expect(WORKBENCH_PROACTIVE_GOAL_STATUS_MAP).toEqual({
      active: "active",
      paused: "paused",
      completed: "completed",
    });
    expect(WORKBENCH_PROACTIVE_TASK_STATUS_MAP).toEqual({
      pending: "queued",
      running: "active",
      paused: "paused",
      completed: "completed",
      failed: "blocked",
    });
  });

  it("defines the phase-0 deep-link inventory", () => {
    expect(WORKBENCH_DEEP_LINK_SUPPORT).toEqual({
      session: { available: true, params: ["sessionId"] },
      project: { available: true, params: ["projectId"] },
      "task-context": { available: false, params: ["taskId"] },
    });
  });

  it("keeps the default refresh strategy and row limit stable", () => {
    expect(WORKBENCH_POLL_INTERVAL_MS).toBe(30_000);
    expect(WORKBENCH_DEFAULT_ROW_LIMIT).toBe(100);
  });

  it("creates empty counts for every unified status", () => {
    expect(createEmptyWorkbenchCounts()).toEqual({
      active: 0,
      paused: 0,
      blocked: 0,
      queued: 0,
      completed: 0,
      archived: 0,
    });
  });

  it("truncates summaries to the phase-0 preview contract", () => {
    const longText = "a".repeat(WORKBENCH_SUMMARY_MAX_CHARS + 25);
    const summary = truncateWorkbenchSummary(longText);
    expect(summary.length).toBeLessThanOrEqual(WORKBENCH_SUMMARY_MAX_CHARS);
    expect(summary.endsWith("…")).toBe(true);
  });
});

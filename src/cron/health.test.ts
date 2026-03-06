import { describe, expect, it } from "vitest";
import { enrichCronJob, summarizeCronJobs } from "./health.js";
import type { CronJob } from "./types.js";

function makeJob(overrides?: Partial<CronJob>): CronJob {
  return {
    id: "job-1",
    name: "Job 1",
    enabled: true,
    createdAtMs: 1,
    updatedAtMs: 1,
    schedule: { kind: "every", everyMs: 60_000, anchorMs: 1 },
    sessionTarget: "isolated",
    wakeMode: "next-heartbeat",
    payload: { kind: "agentTurn", message: "hi" },
    state: {},
    ...overrides,
  };
}

describe("cron health helpers", () => {
  it("enriches jobs with health and last-run metadata", () => {
    const entry = enrichCronJob(
      makeJob({
        state: {
          lastRunAtMs: 10,
          lastStatus: "error",
          lastError: "boom",
          lastSummary: "failed sync",
          lastSessionId: "sess_1",
          lastSessionKey: "agent:main:cron:job-1:run:sess_1",
          lastModel: "gpt-5",
          lastProvider: "openai",
          consecutiveErrors: 2,
        },
      }),
    );

    expect(entry.health).toEqual({
      severity: "degraded",
      running: false,
      autoDisabled: false,
      disabledReason: undefined,
      consecutiveErrors: 2,
      scheduleErrorCount: 0,
      lastStatus: "error",
    });
    expect(entry.lastRun).toMatchObject({
      status: "error",
      error: "boom",
      summary: "failed sync",
      sessionId: "sess_1",
      sessionKey: "agent:main:cron:job-1:run:sess_1",
      model: "gpt-5",
      provider: "openai",
    });
  });

  it("summarizes active, degraded, running, and auto-disabled jobs", () => {
    const summary = summarizeCronJobs([
      makeJob(),
      makeJob({
        id: "job-2",
        state: { runningAtMs: 10 },
      }),
      makeJob({
        id: "job-3",
        enabled: false,
        state: { scheduleErrorCount: 3, lastError: "schedule error: bad expr" },
      }),
    ]);

    expect(summary).toEqual({
      activeJobs: 2,
      disabledJobs: 1,
      degradedJobs: 1,
      runningJobs: 1,
      autoDisabledJobs: 1,
    });
  });
});

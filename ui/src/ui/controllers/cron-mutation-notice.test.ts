import { describe, expect, it, vi } from "vitest";
import {
  addCronMutationNotice,
  parseCronMutationNotice,
  removeCronMutationNotice,
} from "./cron-mutation-notice.js";

describe("cron mutation notices", () => {
  it("parses cron mutation payloads into expiring notices", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-31T12:00:00.000Z"));

    const parsed = parseCronMutationNotice(
      {
        action: "added",
        jobId: "job-1",
        jobName: "Morning brief",
        agentId: "main",
        sessionTarget: "isolated",
        deliveryMode: "announce",
        nextRunAtMs: Date.parse("2026-04-01T00:00:00.000Z"),
      },
      42,
    );

    expect(parsed).toMatchObject({
      id: "cron:42:added:job-1",
      action: "added",
      jobId: "job-1",
      jobName: "Morning brief",
      agentId: "main",
      sessionTarget: "isolated",
      deliveryMode: "announce",
      nextRunAtMs: Date.parse("2026-04-01T00:00:00.000Z"),
    });
    expect(parsed?.expiresAtMs).toBe(Date.parse("2026-03-31T12:00:08.000Z"));
  });

  it("ignores non-mutation cron events", () => {
    expect(parseCronMutationNotice({ action: "started", jobId: "job-1" })).toBeNull();
  });

  it("adds and removes notices by id", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-31T12:00:00.000Z"));

    const entry = parseCronMutationNotice({ action: "removed", jobId: "job-2" }, 9);
    expect(entry).not.toBeNull();

    const queued = addCronMutationNotice([], entry!);
    expect(queued).toHaveLength(1);
    expect(removeCronMutationNotice(queued, entry!.id)).toEqual([]);
  });
});

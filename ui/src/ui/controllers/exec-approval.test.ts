import { describe, expect, it, vi } from "vitest";
import { parseExecApprovalRequested, pruneExecApprovalQueue } from "./exec-approval.js";

describe("parseExecApprovalRequested", () => {
  it("keeps kind and taskId for UI rendering", () => {
    const entry = parseExecApprovalRequested({
      id: "approval-1",
      request: {
        command: "request_escalation execute",
        kind: "permission-escalation",
        taskId: "task-123",
        cwd: "/workspace",
      },
      createdAtMs: 1000,
      expiresAtMs: 5000,
    });

    expect(entry).toMatchObject({
      id: "approval-1",
      request: {
        command: "request_escalation execute",
        kind: "permission-escalation",
        taskId: "task-123",
      },
    });
  });
});

describe("pruneExecApprovalQueue", () => {
  it("drops expired entries and keeps active ones", () => {
    vi.useFakeTimers();
    vi.setSystemTime(2000);

    const queue = [
      {
        id: "expired",
        request: { command: "echo old" },
        createdAtMs: 1000,
        expiresAtMs: 1500,
      },
      {
        id: "active",
        request: { command: "echo new", kind: "permission-escalation", taskId: "task-1" },
        createdAtMs: 1000,
        expiresAtMs: 4000,
      },
    ];

    expect(pruneExecApprovalQueue(queue)).toHaveLength(1);
    vi.useRealTimers();
  });
});

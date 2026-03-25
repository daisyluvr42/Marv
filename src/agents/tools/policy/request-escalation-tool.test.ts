import { afterEach, describe, expect, it, vi } from "vitest";
import { getEscalationManager, resetEscalationManager } from "./permission-escalation.js";
import { createRequestEscalationTool } from "./request-escalation-tool.js";

const { callGatewayToolMock } = vi.hoisted(() => ({
  callGatewayToolMock: vi.fn(),
}));

vi.mock("../gateway.js", () => ({
  callGatewayTool: callGatewayToolMock,
  readGatewayCallOptions: vi.fn().mockReturnValue({}),
}));

describe("request_escalation tool", () => {
  afterEach(() => {
    vi.clearAllMocks();
    resetEscalationManager();
  });

  it("records approved escalation and grants permission", async () => {
    callGatewayToolMock.mockResolvedValue({ decision: "allow-always" });

    const tool = createRequestEscalationTool({
      agentSessionKey: "agent:main:main",
      config: {},
    });

    const result = await tool.execute("call1", {
      requestedLevel: "execute",
      currentLevel: "read",
      reason: "Need to run install script",
      taskId: "task-1",
    });
    const details = result.details as { taskId: string; granted: boolean };

    expect(details.taskId).toBe("task-1");
    expect(details.granted).toBe(true);
    expect((result.details as { approvalId?: string }).approvalId).toMatch(/^esc-/);
    expect(getEscalationManager().checkPermission("task-1", "execute")).toBe(true);
    expect(callGatewayToolMock).toHaveBeenCalledWith(
      "exec.approval.request",
      {},
      expect.objectContaining({
        taskId: "task-1",
        agentId: "main",
      }),
      { expectFinal: true },
    );
  });

  it("denies escalation when approval is rejected", async () => {
    callGatewayToolMock.mockResolvedValue({ decision: "deny" });

    const tool = createRequestEscalationTool({
      agentSessionKey: "agent:main:main",
      config: {},
    });

    const result = await tool.execute("call2", {
      requestedLevel: "admin",
      reason: "Need full access",
      taskId: "task-2",
    });
    const details = result.details as { granted: boolean; decision: string };

    expect(details.decision).toBe("deny");
    expect(details.granted).toBe(false);
    expect(getEscalationManager().checkPermission("task-2", "admin")).toBe(false);
  });
});

import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  loadConfig: vi.fn(),
  listAgentIds: vi.fn(),
  resolveDefaultAgentId: vi.fn(),
  getMemoryStatusSnapshot: vi.fn(),
  getKnowledgeStatusSnapshot: vi.fn(),
  getProactiveStatusSnapshot: vi.fn(),
  getWorkbenchSnapshot: vi.fn(),
}));

vi.mock("../../config/config.js", () => ({
  loadConfig: mocks.loadConfig,
}));

vi.mock("../../../agents/agent-scope.js", () => ({
  listAgentIds: mocks.listAgentIds,
  resolveDefaultAgentId: mocks.resolveDefaultAgentId,
}));

vi.mock("../../../memory/status.js", () => ({
  getMemoryStatusSnapshot: mocks.getMemoryStatusSnapshot,
}));

vi.mock("../../../knowledge/status.js", () => ({
  getKnowledgeStatusSnapshot: mocks.getKnowledgeStatusSnapshot,
}));

vi.mock("../../../proactive/status.js", () => ({
  getProactiveStatusSnapshot: mocks.getProactiveStatusSnapshot,
}));

vi.mock("../../../workbench/snapshot.js", () => ({
  getWorkbenchSnapshot: mocks.getWorkbenchSnapshot,
}));

import { dashboardHandlers } from "./dashboard.js";

describe("dashboardHandlers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.loadConfig.mockReturnValue({});
    mocks.listAgentIds.mockReturnValue(["main", "research"]);
    mocks.resolveDefaultAgentId.mockReturnValue("main");
    mocks.getMemoryStatusSnapshot.mockReturnValue({ agentId: "main", totalItems: 3 });
    mocks.getKnowledgeStatusSnapshot.mockResolvedValue({ agentId: "main", vaultCount: 1 });
    mocks.getProactiveStatusSnapshot.mockResolvedValue({ agentId: "main", pendingEntries: 2 });
    mocks.getWorkbenchSnapshot.mockResolvedValue({ agentId: "main", rows: [] });
  });

  it("returns memory stats for the default agent", async () => {
    const respond = vi.fn();

    await dashboardHandlers["memory.stats"]({
      params: {},
      respond,
    } as never);

    expect(mocks.getMemoryStatusSnapshot).toHaveBeenCalledWith({
      agentId: "main",
      config: {},
    });
    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({ agentId: "main", totalItems: 3 }),
      undefined,
    );
  });

  it("returns knowledge status for an explicit agent", async () => {
    const respond = vi.fn();

    await dashboardHandlers["knowledge.status"]({
      params: { agentId: "Research" },
      respond,
    } as never);

    expect(mocks.getKnowledgeStatusSnapshot).toHaveBeenCalledWith({
      agentId: "research",
      config: {},
    });
    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({ agentId: "main", vaultCount: 1 }),
      undefined,
    );
  });

  it("rejects unknown agents", async () => {
    const respond = vi.fn();

    await dashboardHandlers["proactive.buffer"]({
      params: { agentId: "unknown" },
      respond,
    } as never);

    expect(mocks.getProactiveStatusSnapshot).not.toHaveBeenCalled();
    expect(respond).toHaveBeenCalledWith(
      false,
      undefined,
      expect.objectContaining({
        message: 'unknown agent id "unknown"',
      }),
    );
  });

  it("returns the aggregated workbench snapshot", async () => {
    const respond = vi.fn();

    await dashboardHandlers["workbench.status"]({
      params: { agentId: "main" },
      respond,
    } as never);

    expect(mocks.getWorkbenchSnapshot).toHaveBeenCalledWith({
      agentId: "main",
    });
    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({ agentId: "main", rows: [] }),
      undefined,
    );
  });
});

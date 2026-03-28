import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  resolveWorkspaceAgent: vi.fn(),
  listSoulMemoryItems: vi.fn(),
  querySoulMemoryMulti: vi.fn(),
  resolveSoulScopes: vi.fn(),
}));

vi.mock("./workspace-agent.js", () => ({
  resolveWorkspaceAgent: mocks.resolveWorkspaceAgent,
}));

vi.mock("../../../memory/storage/soul-memory-store.js", () => ({
  listSoulMemoryItems: mocks.listSoulMemoryItems,
  querySoulMemoryMulti: mocks.querySoulMemoryMulti,
}));

vi.mock("../../../agents/memory-soul-scopes.js", () => ({
  resolveSoulScopes: mocks.resolveSoulScopes,
}));

import { memoryHandlers } from "./memory.js";

describe("memoryHandlers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.resolveWorkspaceAgent.mockReturnValue({
      ok: true,
      cfg: {},
      agentId: "main",
    });
    mocks.resolveSoulScopes.mockReturnValue([{ scopeType: "agent", scopeId: "main", weight: 1 }]);
    mocks.listSoulMemoryItems.mockReturnValue([{ id: "mem-1", content: "remember this" }]);
    mocks.querySoulMemoryMulti.mockReturnValue([{ id: "mem-2", score: 0.91 }]);
  });

  it("returns memory.list results for the resolved agent", async () => {
    const respond = vi.fn();

    await memoryHandlers["memory.list"]({
      params: { limit: 20, tier: "palace" },
      respond,
    } as never);

    expect(mocks.listSoulMemoryItems).toHaveBeenCalledWith(
      expect.objectContaining({
        agentId: "main",
        tier: "palace",
        limit: 20,
      }),
    );
    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        agentId: "main",
        items: expect.any(Array),
      }),
      undefined,
    );
  });

  it("uses resolved default scopes when memory.search omits scopes", async () => {
    const respond = vi.fn();

    await memoryHandlers["memory.search"]({
      params: { query: "roadmap" },
      respond,
    } as never);

    expect(mocks.resolveSoulScopes).toHaveBeenCalledWith({
      agentId: "main",
      sessionKey: undefined,
    });
    expect(mocks.querySoulMemoryMulti).toHaveBeenCalledWith(
      expect.objectContaining({
        agentId: "main",
        query: "roadmap",
        scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
        topK: 20,
      }),
    );
    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        agentId: "main",
        query: "roadmap",
      }),
      undefined,
    );
  });

  it("rejects invalid memory.search params", async () => {
    const respond = vi.fn();

    await memoryHandlers["memory.search"]({
      params: {},
      respond,
    } as never);

    expect(mocks.querySoulMemoryMulti).not.toHaveBeenCalled();
    expect(respond).toHaveBeenCalledWith(
      false,
      undefined,
      expect.objectContaining({
        message: expect.stringContaining("invalid memory.search params"),
      }),
    );
  });
});

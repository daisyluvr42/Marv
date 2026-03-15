import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  loadConfig: vi.fn(),
  listAgentIds: vi.fn(),
  resolveAgentWorkspaceDir: vi.fn(),
}));

vi.mock("../../config/config.js", () => ({
  loadConfig: mocks.loadConfig,
}));

vi.mock("../../../agents/agent-scope.js", () => ({
  listAgentIds: mocks.listAgentIds,
  resolveAgentWorkspaceDir: mocks.resolveAgentWorkspaceDir,
}));

import { documentsHandlers } from "./documents.js";

describe("documentsHandlers", () => {
  let workspaceDir = "";

  beforeEach(async () => {
    vi.clearAllMocks();
    workspaceDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-docs-"));
    await fs.writeFile(path.join(workspaceDir, "NOTES.md"), "# Notes\nhello workspace\n", "utf8");
    await fs.writeFile(
      path.join(workspaceDir, "SOUL.md"),
      "# Soul\nshould stay under Agent\n",
      "utf8",
    );
    await fs.mkdir(path.join(workspaceDir, "reports"), { recursive: true });
    await fs.writeFile(
      path.join(workspaceDir, "reports", "daily.txt"),
      "daily summary\nwith details\n",
      "utf8",
    );
    mocks.loadConfig.mockReturnValue({});
    mocks.listAgentIds.mockReturnValue(["main"]);
    mocks.resolveAgentWorkspaceDir.mockReturnValue(workspaceDir);
  });

  afterEach(async () => {
    if (workspaceDir) {
      await fs.rm(workspaceDir, { recursive: true, force: true });
    }
  });

  it("lists workspace documents with roots and recent items", async () => {
    const respond = vi.fn();

    await documentsHandlers["documents.list"]({
      params: {},
      respond,
    } as never);

    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        roots: [
          expect.objectContaining({
            agentId: "main",
            path: workspaceDir,
          }),
        ],
        items: expect.arrayContaining([
          expect.objectContaining({ relativePath: "NOTES.md" }),
          expect.objectContaining({ relativePath: "reports/daily.txt" }),
        ]),
      }),
      undefined,
    );
    const payload = respond.mock.calls[0]?.[1] as {
      items: Array<{ relativePath: string }>;
    };
    expect(payload.items.some((entry) => entry.relativePath === "SOUL.md")).toBe(false);
  });

  it("reads a document by rootId and relative path", async () => {
    const listRespond = vi.fn();
    await documentsHandlers["documents.list"]({
      params: {},
      respond: listRespond,
    } as never);
    const payload = listRespond.mock.calls[0]?.[1] as {
      roots: Array<{ id: string }>;
    };
    const rootId = payload.roots[0]?.id;
    const readRespond = vi.fn();

    await documentsHandlers["documents.read"]({
      params: { rootId, relativePath: "NOTES.md" },
      respond: readRespond,
    } as never);

    expect(readRespond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        rootId,
        relativePath: "NOTES.md",
        content: expect.stringContaining("hello workspace"),
      }),
      undefined,
    );
  });

  it("rejects path traversal for documents.read", async () => {
    const listRespond = vi.fn();
    await documentsHandlers["documents.list"]({
      params: {},
      respond: listRespond,
    } as never);
    const payload = listRespond.mock.calls[0]?.[1] as {
      roots: Array<{ id: string }>;
    };
    const rootId = payload.roots[0]?.id;
    const respond = vi.fn();

    await documentsHandlers["documents.read"]({
      params: { rootId, relativePath: "../secrets.md" },
      respond,
    } as never);

    expect(respond).toHaveBeenCalledWith(
      false,
      undefined,
      expect.objectContaining({
        message: "relativePath must stay within the workspace root",
      }),
    );
  });
});

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../core/config/types.js";

const mocks = vi.hoisted(() => ({
  snapshot: {
    valid: true,
    exists: false,
    config: {} as MarvConfig,
  },
  writeOptions: {},
  writeConfigFile: vi.fn(async () => {}),
  logConfigUpdated: vi.fn(),
  resolveAgentWorkspaceDir: vi.fn(() => "/tmp/marv-p0"),
}));

vi.mock("../core/config/config.js", () => ({
  readConfigFileSnapshotForWrite: async () => ({
    snapshot: mocks.snapshot,
    writeOptions: mocks.writeOptions,
  }),
  writeConfigFile: mocks.writeConfigFile,
}));

vi.mock("../core/config/logging.js", () => ({
  logConfigUpdated: mocks.logConfigUpdated,
}));

vi.mock("../agents/agent-scope.js", () => ({
  resolveAgentWorkspaceDir: mocks.resolveAgentWorkspaceDir,
}));

const { memoryP0SectionCommand, memoryP0ShowCommand, memoryP0SyncCommand } =
  await import("./memory-p0.js");

function makeRuntime() {
  return {
    log: vi.fn(),
    error: vi.fn(),
    exit: vi.fn(),
  } as never;
}

describe("memory P0 commands", () => {
  let workspaceDir = "";

  beforeEach(async () => {
    vi.clearAllMocks();
    workspaceDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-p0-"));
    mocks.resolveAgentWorkspaceDir.mockReturnValue(workspaceDir);
    mocks.snapshot = {
      valid: true,
      exists: true,
      config: {},
    };
  });

  afterEach(async () => {
    if (workspaceDir) {
      await fs.rm(workspaceDir, { recursive: true, force: true });
    }
  });

  it("shows all P0 sections", async () => {
    mocks.snapshot.config = {
      agents: {
        defaults: {
          p0: {
            soul: "Kind and steady",
            identity: "Marv",
            user: "Prefers Chinese",
          },
        },
      },
    };
    const runtime = makeRuntime();

    await memoryP0ShowCommand({}, runtime);

    expect(runtime.log).toHaveBeenCalledWith("P0 Soul:");
    expect(runtime.log).toHaveBeenCalledWith("Kind and steady");
    expect(runtime.log).toHaveBeenCalledWith("P0 Identity:");
    expect(runtime.log).toHaveBeenCalledWith("Marv");
    expect(runtime.log).toHaveBeenCalledWith("P0 User:");
    expect(runtime.log).toHaveBeenCalledWith("Prefers Chinese");
  });

  it("updates a single P0 section through config", async () => {
    const runtime = makeRuntime();

    await memoryP0SectionCommand("soul", "Stay warm and polite.", {}, runtime);

    expect(mocks.writeConfigFile).toHaveBeenCalledWith(
      expect.objectContaining({
        agents: {
          defaults: {
            p0: {
              soul: "Stay warm and polite.",
            },
          },
        },
      }),
      mocks.writeOptions,
    );
    expect(mocks.logConfigUpdated).toHaveBeenCalledWith(runtime);
  });

  it("syncs SOUL/IDENTITY/USER files back into structured P0", async () => {
    await fs.writeFile(path.join(workspaceDir, "SOUL.md"), "Soul file", "utf8");
    await fs.writeFile(path.join(workspaceDir, "IDENTITY.md"), "Identity file", "utf8");
    await fs.writeFile(path.join(workspaceDir, "USER.md"), "User file", "utf8");
    const runtime = makeRuntime();

    await memoryP0SyncCommand({}, runtime);

    expect(mocks.writeConfigFile).toHaveBeenCalledWith(
      expect.objectContaining({
        agents: {
          defaults: {
            p0: {
              soul: "Soul file",
              identity: "Identity file",
              user: "User file",
            },
          },
        },
      }),
      mocks.writeOptions,
    );
  });
});

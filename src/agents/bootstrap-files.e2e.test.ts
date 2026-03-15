import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  clearInternalHooks,
  registerInternalHook,
  type AgentBootstrapHookContext,
} from "../hooks/internal-hooks.js";
import { makeTempWorkspace } from "../test-helpers/workspace.js";
import { writeWorkspaceFile } from "../test-helpers/workspace.js";
import { resolveBootstrapContextForRun, resolveBootstrapFilesForRun } from "./bootstrap-files.js";
import type { WorkspaceBootstrapFile } from "./workspace.js";

function registerExtraBootstrapFileHook() {
  registerInternalHook("agent:bootstrap", (event) => {
    const context = event.context as AgentBootstrapHookContext;
    context.bootstrapFiles = [
      ...context.bootstrapFiles,
      {
        name: "EXTRA.md",
        path: path.join(context.workspaceDir, "EXTRA.md"),
        content: "extra",
        missing: false,
      } as unknown as WorkspaceBootstrapFile,
    ];
  });
}

describe("resolveBootstrapFilesForRun", () => {
  beforeEach(() => clearInternalHooks());
  afterEach(() => clearInternalHooks());

  it("applies bootstrap hook overrides", async () => {
    registerExtraBootstrapFileHook();

    const workspaceDir = await makeTempWorkspace("marv-bootstrap-");
    const files = await resolveBootstrapFilesForRun({ workspaceDir });

    expect(files.some((file) => file.path === path.join(workspaceDir, "EXTRA.md"))).toBe(true);
  });
});

describe("resolveBootstrapContextForRun", () => {
  beforeEach(() => clearInternalHooks());
  afterEach(() => clearInternalHooks());

  it("returns context files for hook-adjusted bootstrap files", async () => {
    registerExtraBootstrapFileHook();

    const workspaceDir = await makeTempWorkspace("marv-bootstrap-");
    const result = await resolveBootstrapContextForRun({
      workspaceDir,
      config: {
        memory: {
          autoRecall: {
            enabled: false,
          },
        },
      },
    });
    const extra = result.contextFiles.find(
      (file) => file.path === path.join(workspaceDir, "EXTRA.md"),
    );

    expect(extra?.content).toBe("extra");
  });

  it("keeps only SOUL and TOOLS in injected context when auto recall is enabled", async () => {
    const workspaceDir = await makeTempWorkspace("marv-bootstrap-");
    await writeWorkspaceFile({
      dir: workspaceDir,
      name: "SOUL.md",
      content: `${"a".repeat(8100)}\n## Long Memory\n\nhidden section`,
    });
    await writeWorkspaceFile({
      dir: workspaceDir,
      name: "TOOLS.md",
      content: "tooling",
    });
    await writeWorkspaceFile({
      dir: workspaceDir,
      name: "USER.md",
      content: "should not inject",
    });

    const result = await resolveBootstrapContextForRun({
      workspaceDir,
      config: {
        memory: {
          autoRecall: {
            enabled: true,
          },
        },
      },
    });

    const basenames = result.contextFiles.map((file) => path.basename(file.path));
    expect(basenames).toContain("SOUL.md");
    expect(basenames).toContain("TOOLS.md");
    expect(basenames).not.toContain("USER.md");
    expect(
      result.contextFiles.find((file) => path.basename(file.path) === "SOUL.md")?.content,
    ).not.toContain("hidden section");
  });

  it("injects structured P0 labels and suppresses raw SOUL/IDENTITY/USER files when configured", async () => {
    const workspaceDir = await makeTempWorkspace("marv-bootstrap-");
    await writeWorkspaceFile({
      dir: workspaceDir,
      name: "SOUL.md",
      content: "stale soul file",
    });
    await writeWorkspaceFile({
      dir: workspaceDir,
      name: "IDENTITY.md",
      content: "stale identity file",
    });
    await writeWorkspaceFile({
      dir: workspaceDir,
      name: "USER.md",
      content: "stale user file",
    });

    const result = await resolveBootstrapContextForRun({
      workspaceDir,
      config: {
        agents: {
          defaults: {
            p0: {
              soul: "Soul from config",
              identity: "Identity from config",
              user: "User from config",
            },
          },
        },
      },
    });

    expect(result.contextFiles).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ path: "P0 Soul", content: "Soul from config" }),
        expect.objectContaining({ path: "P0 Identity", content: "Identity from config" }),
        expect.objectContaining({ path: "P0 User", content: "User from config" }),
      ]),
    );
    const labels = new Set(result.contextFiles.map((file) => file.path));
    expect(labels.has(path.join(workspaceDir, "SOUL.md"))).toBe(false);
    expect(labels.has(path.join(workspaceDir, "IDENTITY.md"))).toBe(false);
    expect(labels.has(path.join(workspaceDir, "USER.md"))).toBe(false);
  });
});

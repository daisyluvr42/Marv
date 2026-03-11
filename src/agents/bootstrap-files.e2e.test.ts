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
    const result = await resolveBootstrapContextForRun({ workspaceDir });
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
      content: "# Soul\n\ncore anchor\n\n## Long Memory\n\nhidden section",
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
});

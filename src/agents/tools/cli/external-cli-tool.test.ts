import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const runCommandWithTimeoutMock = vi.fn();
const detectMock = vi.fn();
const buildInvocationMock = vi.fn();
const parseOutputMock = vi.fn();

vi.mock("../../../process/exec.js", () => ({
  runCommandWithTimeout: (...args: unknown[]) => runCommandWithTimeoutMock(...args),
  runExec: vi.fn(),
}));

vi.mock("./external-cli-adapters.js", () => ({
  normalizeExternalCliId: (value: string | undefined) => {
    const normalized = value?.trim().toLowerCase();
    if (
      normalized === "codex" ||
      normalized === "claude" ||
      normalized === "aider" ||
      normalized === "gemini"
    ) {
      return normalized;
    }
    return null;
  },
  listExternalCliAdapterIds: () => ["codex", "claude", "aider", "gemini"],
  getExternalCliAdapter: () => ({
    id: "codex",
    command: "codex",
    detect: (...args: unknown[]) => detectMock(...args),
    buildInvocation: (...args: unknown[]) => buildInvocationMock(...args),
    parseOutput: (...args: unknown[]) => parseOutputMock(...args),
  }),
}));

import { createExternalCliTool } from "./external-cli-tool.js";

function createConfig(overrides: Record<string, unknown> = {}) {
  return {
    tools: {
      externalCli: {
        enabled: true,
        ...overrides,
      },
    },
  } as never;
}

describe("external_cli tool", () => {
  let workspaceDir: string;

  beforeEach(async () => {
    workspaceDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-external-cli-test-"));
    runCommandWithTimeoutMock.mockReset();
    detectMock.mockReset();
    buildInvocationMock.mockReset();
    parseOutputMock.mockReset();
    buildInvocationMock.mockReturnValue({
      command: "codex",
      args: ["exec", "Fix the bug"],
    });
    parseOutputMock.mockImplementation((stdout: string, stderr: string) => ({
      text: stdout.trim() || stderr.trim(),
      raw: stdout.trim(),
    }));
  });

  afterEach(async () => {
    await fs.rm(workspaceDir, { recursive: true, force: true }).catch(() => {});
  });

  it("returns null when the feature is disabled", () => {
    expect(
      createExternalCliTool({
        config: { tools: { externalCli: { enabled: false } } } as never,
        workspaceDir,
      }),
    ).toBeNull();
  });

  it("returns null in sandboxed sessions", () => {
    expect(
      createExternalCliTool({
        config: createConfig(),
        workspaceDir,
        sandboxed: true,
      }),
    ).toBeNull();
  });

  it("returns not_configured when no preferred or available CLI is known", async () => {
    const tool = createExternalCliTool({
      config: createConfig({ defaultCli: undefined, availableCli: undefined }),
      workspaceDir,
    });
    expect(tool).not.toBeNull();
    const result = await tool!.execute("call1", { task: "Fix the bug" });

    expect(result.details).toMatchObject({
      status: "not_configured",
    });
    expect(detectMock).not.toHaveBeenCalled();
  });

  it("runs an explicitly requested CLI even before brands are remembered", async () => {
    detectMock.mockResolvedValue(true);
    runCommandWithTimeoutMock.mockResolvedValue({
      stdout: "delegated successfully",
      stderr: "",
      code: 0,
      signal: null,
      killed: false,
      termination: "exit",
    });

    const tool = createExternalCliTool({
      config: createConfig({ defaultCli: undefined, availableCli: undefined }),
      workspaceDir,
    });
    expect(tool).not.toBeNull();
    const result = await tool!.execute("call2", {
      cli: "codex",
      task: "Fix the bug",
    });

    expect(buildInvocationMock).toHaveBeenCalledWith({
      task: "Fix the bug",
      model: undefined,
      override: undefined,
    });
    expect(runCommandWithTimeoutMock).toHaveBeenCalledWith(["codex", "exec", "Fix the bug"], {
      cwd: workspaceDir,
      timeoutMs: 300_000,
      input: undefined,
      env: undefined,
    });
    expect(result.details).toMatchObject({
      status: "ok",
      cli: "codex",
      output: "delegated successfully",
      workdir: workspaceDir,
    });
  });

  it("returns not_available when the selected CLI cannot be detected", async () => {
    detectMock.mockResolvedValue(false);

    const tool = createExternalCliTool({
      config: createConfig({ defaultCli: "codex", availableCli: ["codex"] }),
      workspaceDir,
    });
    expect(tool).not.toBeNull();
    const result = await tool!.execute("call3", {
      task: "Fix the bug",
    });

    expect(result.details).toMatchObject({
      status: "not_available",
      cli: "codex",
    });
    expect(runCommandWithTimeoutMock).not.toHaveBeenCalled();
  });

  it("marks quota exhaustion so the main agent can take over", async () => {
    detectMock.mockResolvedValue(true);
    runCommandWithTimeoutMock.mockResolvedValue({
      stdout: "",
      stderr: "usage limit reached after partial edits",
      code: 1,
      signal: null,
      killed: false,
      termination: "exit",
    });
    parseOutputMock.mockReturnValue({
      text: "usage limit reached after partial edits",
      raw: "usage limit reached after partial edits",
    });

    const tool = createExternalCliTool({
      config: createConfig({ defaultCli: "codex", availableCli: ["codex"] }),
      workspaceDir,
    });
    expect(tool).not.toBeNull();
    const result = await tool!.execute("call4", {
      task: "Fix the bug",
      kind: "general",
    });

    expect(result.details).toMatchObject({
      status: "quota_exhausted",
      cli: "codex",
      exitReason: "quota_exhausted",
    });
  });
});

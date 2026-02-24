import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runCommandWithTimeout } from "../process/exec.js";
import {
  __testing,
  ensureWorkspaceSnapshotBeforeMutation,
  listWorkspaceSnapshots,
  revertWorkspaceToSnapshot,
} from "./workspace-rollback.js";

type GitResult = Awaited<ReturnType<typeof runCommandWithTimeout>>;

let workspaceDir = "";
let stateDir = "";
let prevStateDir: string | undefined;
let prevLegacyStateDir: string | undefined;

async function runGit(args: string[], cwd: string = workspaceDir): Promise<GitResult> {
  return await runCommandWithTimeout(["git", "-C", cwd, ...args], {
    timeoutMs: 15_000,
  });
}

async function expectGitOk(args: string[], cwd?: string): Promise<GitResult> {
  const result = await runGit(args, cwd);
  if (result.code !== 0) {
    throw new Error(
      `git ${args.join(" ")} failed (code=${String(result.code)}): ${result.stdout}\n${result.stderr}`,
    );
  }
  return result;
}

async function initRepoWithBaseline(content = "hello\n"): Promise<void> {
  await expectGitOk(["init"]);
  await expectGitOk(["config", "user.name", "Test User"]);
  await expectGitOk(["config", "user.email", "test@example.com"]);
  await fs.writeFile(path.join(workspaceDir, "note.txt"), content, "utf-8");
  await expectGitOk(["add", "note.txt"]);
  await expectGitOk(["commit", "-m", "init"]);
}

beforeEach(async () => {
  workspaceDir = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-rollback-workspace-"));
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-rollback-state-"));
  prevStateDir = process.env.MARV_STATE_DIR;
  prevLegacyStateDir = process.env.OPENCLAW_STATE_DIR;
  process.env.MARV_STATE_DIR = stateDir;
  process.env.OPENCLAW_STATE_DIR = stateDir;
  __testing.resetWorkspaceSnapshotStateForTest();
});

afterEach(async () => {
  if (prevStateDir === undefined) {
    delete process.env.MARV_STATE_DIR;
  } else {
    process.env.MARV_STATE_DIR = prevStateDir;
  }
  if (prevLegacyStateDir === undefined) {
    delete process.env.OPENCLAW_STATE_DIR;
  } else {
    process.env.OPENCLAW_STATE_DIR = prevLegacyStateDir;
  }
  __testing.resetWorkspaceSnapshotStateForTest();
  await fs.rm(workspaceDir, { recursive: true, force: true });
  await fs.rm(stateDir, { recursive: true, force: true });
});

describe("workspace rollback snapshots", () => {
  it("creates a head snapshot when the repo is clean", async () => {
    await initRepoWithBaseline("clean\n");

    const outcome = await ensureWorkspaceSnapshotBeforeMutation({
      workspaceDir,
      sessionKey: "agent:main:main",
      toolName: "write",
    });

    expect(outcome.status).toBe("created");
    if (outcome.status !== "created") {
      return;
    }
    expect(outcome.snapshot.mode).toBe("head");
    const head = (await expectGitOk(["rev-parse", "HEAD"])).stdout.trim();
    expect(outcome.snapshot.commit).toBe(head);

    const listed = await listWorkspaceSnapshots({ workspaceDir });
    expect(listed).toHaveLength(1);
    expect(listed[0]?.commit).toBe(head);
  });

  it("creates a commit snapshot when the repo is dirty", async () => {
    await initRepoWithBaseline("before\n");
    await fs.writeFile(path.join(workspaceDir, "note.txt"), "dirty-before-tool\n", "utf-8");

    const outcome = await ensureWorkspaceSnapshotBeforeMutation({
      workspaceDir,
      sessionKey: "agent:main:main",
      toolName: "edit",
    });

    expect(outcome.status).toBe("created");
    if (outcome.status !== "created") {
      return;
    }
    expect(outcome.snapshot.mode).toBe("commit");

    const status = await expectGitOk(["status", "--porcelain"]);
    expect(status.stdout.trim()).toBe("");

    const log = await expectGitOk(["log", "-1", "--pretty=%B"]);
    expect(log.stdout).toContain("Auto-save before AI action");
  });

  it("skips creating a second snapshot inside the window", async () => {
    await initRepoWithBaseline("window\n");

    const first = await ensureWorkspaceSnapshotBeforeMutation({
      workspaceDir,
      sessionKey: "agent:main:main",
      toolName: "write",
      windowMs: 60_000,
    });
    const second = await ensureWorkspaceSnapshotBeforeMutation({
      workspaceDir,
      sessionKey: "agent:main:main",
      toolName: "write",
      windowMs: 60_000,
    });

    expect(first.status).toBe("created");
    expect(second).toEqual({
      status: "skipped",
      reason: "snapshot_window",
    });
  });

  it("reverts workspace to the latest snapshot commit", async () => {
    await initRepoWithBaseline("one\n");
    await fs.writeFile(path.join(workspaceDir, "note.txt"), "two\n", "utf-8");
    const snap = await ensureWorkspaceSnapshotBeforeMutation({
      workspaceDir,
      sessionKey: "agent:main:main",
      toolName: "apply_patch",
      windowMs: 0,
    });
    expect(snap.status).toBe("created");

    await fs.writeFile(path.join(workspaceDir, "note.txt"), "three\n", "utf-8");
    await fs.writeFile(path.join(workspaceDir, "temp.txt"), "untracked\n", "utf-8");

    const reverted = await revertWorkspaceToSnapshot({ workspaceDir });
    expect(reverted.ok).toBe(true);
    if (!reverted.ok) {
      return;
    }
    const note = await fs.readFile(path.join(workspaceDir, "note.txt"), "utf-8");
    expect(note).toBe("two\n");

    const status = await expectGitOk(["status", "--porcelain"]);
    expect(status.stdout.trim()).toBe("");
  });
});

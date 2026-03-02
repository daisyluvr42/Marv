import crypto from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { resolveStateDir } from "../config/paths.js";
import { runCommandWithTimeout } from "../process/exec.js";

const SNAPSHOT_TOOL_NAMES = new Set(["write", "edit", "apply_patch"]);
const DEFAULT_SNAPSHOT_WINDOW_MS = 5 * 60 * 1000;
const DEFAULT_GIT_TIMEOUT_MS = 15_000;
const SNAPSHOT_LOG_SEGMENTS = ["rollback", "workspace-snapshots.jsonl"] as const;

type SnapshotMode = "head" | "commit";

export type WorkspaceSnapshotRecord = {
  id: string;
  ts: number;
  workspaceDir: string;
  repoRoot: string;
  sessionKey?: string;
  toolName: string;
  commit: string;
  mode: SnapshotMode;
  message: string;
};

export type EnsureWorkspaceSnapshotResult =
  | {
      status: "created";
      snapshot: WorkspaceSnapshotRecord;
    }
  | {
      status: "skipped";
      reason:
        | "non_mutating_tool"
        | "workspace_missing"
        | "not_git_repo"
        | "snapshot_window"
        | "git_error";
    };

export type RevertWorkspaceToSnapshotResult =
  | {
      ok: true;
      snapshot: WorkspaceSnapshotRecord;
      cleaned: boolean;
    }
  | {
      ok: false;
      code:
        | "workspace_missing"
        | "not_git_repo"
        | "snapshot_not_found"
        | "invalid_ref"
        | "reset_failed";
      message: string;
    };

const lastSnapshotByRepo = new Map<string, number>();

function resolveSnapshotLogPath(): string {
  const stateDir = resolveStateDir(process.env, os.homedir);
  return path.join(stateDir, ...SNAPSHOT_LOG_SEGMENTS);
}

function normalizeToolName(toolName: string): string {
  return toolName.trim().toLowerCase();
}

function normalizeWorkspaceDir(workspaceDir?: string | null): string | null {
  const trimmed = workspaceDir?.trim();
  if (!trimmed) {
    return null;
  }
  return path.resolve(trimmed);
}

function shouldSnapshotTool(toolName: string): boolean {
  return SNAPSHOT_TOOL_NAMES.has(toolName);
}

function buildSnapshotMessage(ts: number): string {
  return `Auto-save before AI action ${new Date(ts).toISOString()}`;
}

async function runGit(params: { cwd: string; args: string[]; timeoutMs?: number }): Promise<{
  ok: boolean;
  stdout: string;
  stderr: string;
}> {
  const result = await runCommandWithTimeout(["git", "-C", params.cwd, ...params.args], {
    timeoutMs: params.timeoutMs ?? DEFAULT_GIT_TIMEOUT_MS,
  });
  return {
    ok: result.code === 0,
    stdout: result.stdout,
    stderr: result.stderr,
  };
}

async function resolveGitRepoRoot(workspaceDir: string): Promise<string | null> {
  const repo = await runGit({
    cwd: workspaceDir,
    args: ["rev-parse", "--show-toplevel"],
    timeoutMs: 3_000,
  });
  if (!repo.ok) {
    return null;
  }
  const root = repo.stdout.trim();
  if (!root) {
    return null;
  }
  return path.resolve(root);
}

async function readSnapshotLog(): Promise<WorkspaceSnapshotRecord[]> {
  const logPath = resolveSnapshotLogPath();
  let raw = "";
  try {
    raw = await fs.readFile(logPath, "utf-8");
  } catch {
    return [];
  }

  const records: WorkspaceSnapshotRecord[] = [];
  for (const line of raw.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    try {
      const parsed = JSON.parse(trimmed) as Partial<WorkspaceSnapshotRecord>;
      if (
        typeof parsed.id !== "string" ||
        typeof parsed.ts !== "number" ||
        typeof parsed.workspaceDir !== "string" ||
        typeof parsed.repoRoot !== "string" ||
        typeof parsed.toolName !== "string" ||
        typeof parsed.commit !== "string" ||
        typeof parsed.mode !== "string" ||
        typeof parsed.message !== "string"
      ) {
        continue;
      }
      const mode = parsed.mode === "commit" ? "commit" : parsed.mode === "head" ? "head" : null;
      if (!mode) {
        continue;
      }
      records.push({
        id: parsed.id,
        ts: parsed.ts,
        workspaceDir: path.resolve(parsed.workspaceDir),
        repoRoot: path.resolve(parsed.repoRoot),
        sessionKey: typeof parsed.sessionKey === "string" ? parsed.sessionKey : undefined,
        toolName: normalizeToolName(parsed.toolName),
        commit: parsed.commit,
        mode,
        message: parsed.message,
      });
    } catch {
      // Ignore malformed lines to keep rollback fail-open.
    }
  }
  return records;
}

async function appendSnapshotLog(record: WorkspaceSnapshotRecord): Promise<void> {
  const logPath = resolveSnapshotLogPath();
  await fs.mkdir(path.dirname(logPath), { recursive: true });
  await fs.appendFile(logPath, `${JSON.stringify(record)}\n`, "utf-8");
}

async function resolveLastSnapshotTs(repoRoot: string): Promise<number | null> {
  const cached = lastSnapshotByRepo.get(repoRoot);
  if (typeof cached === "number") {
    return cached;
  }
  const snapshots = await readSnapshotLog();
  for (let i = snapshots.length - 1; i >= 0; i -= 1) {
    const entry = snapshots[i];
    if (entry?.repoRoot === repoRoot) {
      lastSnapshotByRepo.set(repoRoot, entry.ts);
      return entry.ts;
    }
  }
  return null;
}

async function resolveHeadCommit(repoRoot: string): Promise<string | null> {
  const head = await runGit({
    cwd: repoRoot,
    args: ["rev-parse", "HEAD"],
    timeoutMs: 3_000,
  });
  if (!head.ok) {
    return null;
  }
  const commit = head.stdout.trim();
  return commit || null;
}

export async function ensureWorkspaceSnapshotBeforeMutation(params: {
  workspaceDir?: string | null;
  sessionKey?: string;
  toolName: string;
  windowMs?: number;
}): Promise<EnsureWorkspaceSnapshotResult> {
  const toolName = normalizeToolName(params.toolName);
  if (!shouldSnapshotTool(toolName)) {
    return { status: "skipped", reason: "non_mutating_tool" };
  }

  const workspaceDir = normalizeWorkspaceDir(params.workspaceDir);
  if (!workspaceDir) {
    return { status: "skipped", reason: "workspace_missing" };
  }

  const repoRoot = await resolveGitRepoRoot(workspaceDir);
  if (!repoRoot) {
    return { status: "skipped", reason: "not_git_repo" };
  }

  const now = Date.now();
  const windowMs = Math.max(0, Math.floor(params.windowMs ?? DEFAULT_SNAPSHOT_WINDOW_MS));
  const lastSnapshotTs = await resolveLastSnapshotTs(repoRoot);
  if (lastSnapshotTs != null && now - lastSnapshotTs < windowMs) {
    return { status: "skipped", reason: "snapshot_window" };
  }

  const status = await runGit({
    cwd: repoRoot,
    args: ["status", "--porcelain", "--untracked-files=all"],
    timeoutMs: 5_000,
  });
  if (!status.ok) {
    return { status: "skipped", reason: "git_error" };
  }

  const message = buildSnapshotMessage(now);
  let mode: SnapshotMode = "head";

  if (status.stdout.trim().length > 0) {
    const added = await runGit({
      cwd: repoRoot,
      args: ["add", "-A"],
      timeoutMs: 10_000,
    });
    if (!added.ok) {
      return { status: "skipped", reason: "git_error" };
    }

    const committed = await runGit({
      cwd: repoRoot,
      args: [
        "-c",
        "user.name=Marv Auto Save",
        "-c",
        "user.email=autosave@marv.local",
        "commit",
        "--no-verify",
        "-m",
        message,
      ],
      timeoutMs: 20_000,
    });
    if (!committed.ok) {
      const failureText = `${committed.stdout}\n${committed.stderr}`.toLowerCase();
      if (!failureText.includes("nothing to commit")) {
        return { status: "skipped", reason: "git_error" };
      }
    } else {
      mode = "commit";
    }
  }

  const commit = await resolveHeadCommit(repoRoot);
  if (!commit) {
    return { status: "skipped", reason: "git_error" };
  }

  const snapshot: WorkspaceSnapshotRecord = {
    id: `wsnap_${crypto.randomUUID().replace(/-/g, "")}`,
    ts: now,
    workspaceDir,
    repoRoot,
    sessionKey: params.sessionKey?.trim() || undefined,
    toolName,
    commit,
    mode,
    message,
  };

  try {
    await appendSnapshotLog(snapshot);
    lastSnapshotByRepo.set(repoRoot, snapshot.ts);
  } catch {
    return { status: "skipped", reason: "git_error" };
  }

  return {
    status: "created",
    snapshot,
  };
}

export async function listWorkspaceSnapshots(params: {
  workspaceDir?: string | null;
  limit?: number;
}): Promise<WorkspaceSnapshotRecord[]> {
  const workspaceDir = normalizeWorkspaceDir(params.workspaceDir);
  if (!workspaceDir) {
    return [];
  }
  const repoRoot = await resolveGitRepoRoot(workspaceDir);
  if (!repoRoot) {
    return [];
  }

  const limit = Math.max(1, Math.min(200, Math.floor(params.limit ?? 20)));
  const snapshots = await readSnapshotLog();
  return snapshots
    .filter((entry) => entry.repoRoot === repoRoot)
    .toSorted((a, b) => b.ts - a.ts)
    .slice(0, limit);
}

function buildSnapshotFromRef(params: {
  commit: string;
  repoRoot: string;
  workspaceDir: string;
  ts?: number;
  toolName?: string;
  message?: string;
}): WorkspaceSnapshotRecord {
  return {
    id: `wsnap_${crypto.randomUUID().replace(/-/g, "")}`,
    ts: params.ts ?? Date.now(),
    workspaceDir: params.workspaceDir,
    repoRoot: params.repoRoot,
    toolName: params.toolName ?? "revert",
    commit: params.commit,
    mode: "head",
    message: params.message ?? "Manual snapshot reference",
  };
}

export async function revertWorkspaceToSnapshot(params: {
  workspaceDir?: string | null;
  snapshotRef?: string;
}): Promise<RevertWorkspaceToSnapshotResult> {
  const workspaceDir = normalizeWorkspaceDir(params.workspaceDir);
  if (!workspaceDir) {
    return {
      ok: false,
      code: "workspace_missing",
      message: "Workspace is missing.",
    };
  }
  const repoRoot = await resolveGitRepoRoot(workspaceDir);
  if (!repoRoot) {
    return {
      ok: false,
      code: "not_git_repo",
      message: "Workspace is not inside a git repository.",
    };
  }

  let snapshot: WorkspaceSnapshotRecord | null = null;
  const requestedRef = params.snapshotRef?.trim();

  if (requestedRef) {
    const verify = await runGit({
      cwd: repoRoot,
      args: ["rev-parse", "--verify", `${requestedRef}^{commit}`],
      timeoutMs: 4_000,
    });
    if (!verify.ok) {
      return {
        ok: false,
        code: "invalid_ref",
        message: `Snapshot ref not found: ${requestedRef}`,
      };
    }
    const commit = verify.stdout.trim();
    snapshot = buildSnapshotFromRef({
      commit,
      repoRoot,
      workspaceDir,
      message: `Manual ref ${requestedRef}`,
    });
  } else {
    const latest = (await listWorkspaceSnapshots({ workspaceDir, limit: 1 }))[0];
    if (!latest) {
      return {
        ok: false,
        code: "snapshot_not_found",
        message: "No workspace snapshot is available yet.",
      };
    }
    snapshot = latest;
  }

  const reset = await runGit({
    cwd: repoRoot,
    args: ["reset", "--hard", snapshot.commit],
    timeoutMs: 20_000,
  });
  if (!reset.ok) {
    return {
      ok: false,
      code: "reset_failed",
      message: "git reset --hard failed.",
    };
  }

  const cleaned = await runGit({
    cwd: repoRoot,
    args: ["clean", "-fd"],
    timeoutMs: 10_000,
  });

  return {
    ok: true,
    snapshot,
    cleaned: cleaned.ok,
  };
}

export const __testing = {
  DEFAULT_SNAPSHOT_WINDOW_MS,
  SNAPSHOT_LOG_SEGMENTS,
  buildSnapshotMessage,
  normalizeWorkspaceDir,
  resolveSnapshotLogPath,
  shouldSnapshotTool,
  resetWorkspaceSnapshotStateForTest: () => {
    lastSnapshotByRepo.clear();
  },
};

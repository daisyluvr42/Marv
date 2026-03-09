import { createHash } from "node:crypto";
import path from "node:path";
import { resolveStateDir } from "../../core/config/paths.js";
import { runCommandWithTimeout } from "../../process/exec.js";
import { acquireFileLock } from "../file-lock.js";
import { readJsonFile, writeJsonAtomic } from "../json-files.js";
import { readPackageVersion } from "../package-json.js";

type CommandRunner = (
  argv: string[],
  options: { cwd?: string; timeoutMs?: number },
) => Promise<{ stdout: string; stderr: string; code: number | null }>;

export type DeployRevisionSnapshot = {
  root: string;
  sha: string | null;
  branch: string | null;
  upstream: string | null;
  version: string | null;
  recordedAt: string;
};

export type DeployAttemptStatus = "pending" | "ok" | "error" | "skipped";

export type DeployAttemptSnapshot = {
  status: DeployAttemptStatus;
  startedAt: string;
  finishedAt?: string;
  trigger: "manual" | "cron" | "startup" | "rollback";
  fromSha?: string | null;
  targetSha?: string | null;
  root: string;
  reason?: string;
  rolledBackToSha?: string | null;
};

export type DeployRollbackSnapshot = {
  status: "ok" | "error";
  startedAt: string;
  finishedAt?: string;
  fromSha?: string | null;
  toSha?: string | null;
  reason?: string;
};

export type DeployState = {
  version: 1;
  root: string;
  current?: DeployRevisionSnapshot;
  lastKnownGood?: DeployRevisionSnapshot;
  lastAttempt?: DeployAttemptSnapshot;
  lastRollback?: DeployRollbackSnapshot;
};

const DEFAULT_TIMEOUT_MS = 3_000;
const DEPLOY_STATE_VERSION = 1 as const;
const LOCK_OPTIONS = {
  retries: {
    retries: 20,
    factor: 1.4,
    minTimeout: 50,
    maxTimeout: 500,
    randomize: true,
  },
  stale: 60_000,
} as const;

function normalizeRoot(root: string): string {
  return path.resolve(root);
}

function rootHash(root: string): string {
  return createHash("sha1").update(normalizeRoot(root)).digest("hex").slice(0, 12);
}

export function resolveDeployStatePath(root: string): string {
  return path.join(resolveStateDir(), "update-deploy", `${rootHash(root)}.json`);
}

async function withDeployStateLock<T>(root: string, fn: () => Promise<T>): Promise<T> {
  const lock = await acquireFileLock(resolveDeployStatePath(root), LOCK_OPTIONS);
  try {
    return await fn();
  } finally {
    await lock.release();
  }
}

async function runGitCommand(
  argv: string[],
  opts: { cwd: string; timeoutMs: number; runCommand?: CommandRunner },
): Promise<{ stdout: string; stderr: string; code: number | null }> {
  if (opts.runCommand) {
    return await opts.runCommand(argv, { cwd: opts.cwd, timeoutMs: opts.timeoutMs });
  }
  const res = await runCommandWithTimeout(argv, {
    cwd: opts.cwd,
    timeoutMs: opts.timeoutMs,
  });
  return { stdout: res.stdout, stderr: res.stderr, code: res.code };
}

async function readStateUnlocked(root: string): Promise<DeployState | null> {
  const parsed = await readJsonFile<DeployState>(resolveDeployStatePath(root));
  if (!parsed || parsed.version !== DEPLOY_STATE_VERSION) {
    return null;
  }
  return parsed;
}

async function writeStateUnlocked(root: string, state: DeployState): Promise<void> {
  await writeJsonAtomic(resolveDeployStatePath(root), state, { mode: 0o600 });
}

function buildEmptyState(root: string): DeployState {
  return {
    version: DEPLOY_STATE_VERSION,
    root: normalizeRoot(root),
  };
}

export async function readDeployState(params: { root: string }): Promise<DeployState | null> {
  return await withDeployStateLock(params.root, async () => {
    return await readStateUnlocked(params.root);
  });
}

export async function updateDeployState(
  params: { root: string },
  mutator: (state: DeployState) => void,
): Promise<DeployState> {
  return await withDeployStateLock(params.root, async () => {
    const root = normalizeRoot(params.root);
    const state = (await readStateUnlocked(root)) ?? buildEmptyState(root);
    mutator(state);
    state.root = root;
    state.version = DEPLOY_STATE_VERSION;
    await writeStateUnlocked(root, state);
    return state;
  });
}

export async function snapshotDeployRevision(params: {
  root: string;
  timeoutMs?: number;
  runCommand?: CommandRunner;
}): Promise<DeployRevisionSnapshot | null> {
  const root = normalizeRoot(params.root);
  const timeoutMs = params.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  const [shaRes, branchRes, upstreamRes, version] = await Promise.all([
    runGitCommand(["git", "-C", root, "rev-parse", "HEAD"], {
      cwd: root,
      timeoutMs,
      runCommand: params.runCommand,
    }),
    runGitCommand(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"], {
      cwd: root,
      timeoutMs,
      runCommand: params.runCommand,
    }),
    runGitCommand(
      ["git", "-C", root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
      {
        cwd: root,
        timeoutMs,
        runCommand: params.runCommand,
      },
    ).catch(() => ({ stdout: "", stderr: "", code: 1 })),
    readPackageVersion(root),
  ]);

  if (shaRes.code !== 0) {
    return null;
  }

  const normalize = (value: string) => {
    const trimmed = value.trim();
    return trimmed ? trimmed : null;
  };

  return {
    root,
    sha: normalize(shaRes.stdout),
    branch: normalize(branchRes.stdout),
    upstream: normalize(upstreamRes.stdout),
    version,
    recordedAt: new Date().toISOString(),
  };
}

export async function markDeployHealthy(params: {
  root: string;
  timeoutMs?: number;
  runCommand?: CommandRunner;
}): Promise<DeployState | null> {
  const revision = await snapshotDeployRevision(params);
  if (!revision?.sha) {
    return null;
  }
  return await updateDeployState({ root: params.root }, (state) => {
    state.current = revision;
    state.lastKnownGood = revision;
    if (state.lastAttempt?.status === "pending" && state.lastAttempt.targetSha === revision.sha) {
      state.lastAttempt = {
        ...state.lastAttempt,
        status: "ok",
        finishedAt: new Date().toISOString(),
      };
    }
  });
}

export async function beginDeployAttempt(params: {
  root: string;
  trigger: DeployAttemptSnapshot["trigger"];
  fromSha?: string | null;
  targetSha?: string | null;
}): Promise<DeployState> {
  return await updateDeployState({ root: params.root }, (state) => {
    state.lastAttempt = {
      status: "pending",
      startedAt: new Date().toISOString(),
      trigger: params.trigger,
      fromSha: params.fromSha ?? null,
      targetSha: params.targetSha ?? null,
      root: normalizeRoot(params.root),
    };
  });
}

export async function finishDeployAttempt(params: {
  root: string;
  status: DeployAttemptStatus;
  reason?: string;
  rolledBackToSha?: string | null;
}): Promise<DeployState> {
  return await updateDeployState({ root: params.root }, (state) => {
    state.lastAttempt = {
      ...(state.lastAttempt ?? {
        startedAt: new Date().toISOString(),
        trigger: "manual",
        root: normalizeRoot(params.root),
      }),
      status: params.status,
      finishedAt: new Date().toISOString(),
      reason: params.reason,
      rolledBackToSha: params.rolledBackToSha ?? null,
    };
  });
}

export async function recordDeployRollback(params: {
  root: string;
  status: "ok" | "error";
  fromSha?: string | null;
  toSha?: string | null;
  reason?: string;
}): Promise<DeployState> {
  return await updateDeployState({ root: params.root }, (state) => {
    state.lastRollback = {
      status: params.status,
      startedAt: state.lastRollback?.startedAt ?? new Date().toISOString(),
      finishedAt: new Date().toISOString(),
      fromSha: params.fromSha ?? null,
      toSha: params.toSha ?? null,
      reason: params.reason,
    };
  });
}

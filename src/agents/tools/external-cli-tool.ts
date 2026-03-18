import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../core/config/config.js";
import type {
  ExternalCliAdapterId,
  ExternalCliToolsConfig,
} from "../../core/config/types.tools.js";
import { runCommandWithTimeout, runExec } from "../../process/exec.js";
import { assertSandboxPath } from "../sandbox/sandbox-paths.js";
import { optionalStringEnum } from "../schema/typebox.js";
import { resolveWorkspaceRoot } from "../workspace-dir.js";
import type { AnyAgentTool } from "./common.js";
import { jsonResult, readNumberParam, readStringParam } from "./common.js";
import { getExternalCliAdapter, normalizeExternalCliId } from "./external-cli-adapters.js";

const OUTPUT_LIMIT = 32_000;
const STDERR_LIMIT = 8_000;
const GIT_DIFF_LIMIT = 16_000;
const DEFAULT_TIMEOUT_SECONDS = 300;

const ExternalCliToolSchema = Type.Object({
  cli: optionalStringEnum(["codex", "claude", "aider", "gemini"] as const, {
    description: "External CLI to use. Omit to resolve from preference or availability.",
  }),
  task: Type.String({
    description: "The full task to delegate to the external CLI.",
  }),
  workdir: Type.Optional(
    Type.String({
      description: "Workspace-relative or in-workspace absolute working directory.",
    }),
  ),
  kind: optionalStringEnum(["coding", "research", "ops", "general"] as const, {
    description: "Task category hint for adapter behavior and result handling.",
  }),
  model: Type.Optional(Type.String()),
  timeoutSeconds: Type.Optional(Type.Number({ minimum: 10 })),
  captureGitDiff: Type.Optional(Type.Boolean()),
  isolate: Type.Optional(Type.Boolean()),
});

type ExternalCliResultStatus =
  | "ok"
  | "quota_exhausted"
  | "timeout"
  | "error"
  | "not_available"
  | "not_configured";

type GitStatusEntry = {
  code: string;
  path: string;
};

type BaselineSnapshotEntry = {
  relativePath: string;
  snapshotPath: string;
};

type GitBaseline = {
  repoRoot: string;
  snapshotRoot: string;
  emptyFilePath: string;
  entries: Map<string, BaselineSnapshotEntry>;
};

type GitDiffSummary = {
  filesChanged: string[];
  gitDiff?: string;
};

function resolveExternalCliConfig(cfg?: MarvConfig): ExternalCliToolsConfig | undefined {
  return cfg?.tools?.externalCli;
}

function isExternalCliEnabled(cfg?: MarvConfig): boolean {
  return resolveExternalCliConfig(cfg)?.enabled === true;
}

function normalizeAvailableCli(values: ExternalCliAdapterId[] | undefined): ExternalCliAdapterId[] {
  const out: ExternalCliAdapterId[] = [];
  const seen = new Set<ExternalCliAdapterId>();
  for (const value of values ?? []) {
    if (seen.has(value)) {
      continue;
    }
    seen.add(value);
    out.push(value);
  }
  return out;
}

function truncate(value: string | undefined, limit: number): string | undefined {
  if (!value) {
    return undefined;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  if (trimmed.length <= limit) {
    return trimmed;
  }
  return `${trimmed.slice(0, Math.max(0, limit - 12))}\n...[truncated]`;
}

function detectQuotaExhausted(text: string): boolean {
  const normalized = text.toLowerCase();
  return (
    normalized.includes("quota exceeded") ||
    normalized.includes("quota exhausted") ||
    normalized.includes("usage limit reached") ||
    normalized.includes("credit balance is too low") ||
    normalized.includes("credits exhausted") ||
    normalized.includes("out of credits")
  );
}

async function resolveWorkdir(params: {
  workspaceRoot: string;
  requested?: string;
}): Promise<string> {
  const requested = params.requested?.trim();
  if (!requested) {
    return params.workspaceRoot;
  }
  const resolved = await assertSandboxPath({
    filePath: requested,
    cwd: params.workspaceRoot,
    root: params.workspaceRoot,
  });
  const stat = await fs.stat(resolved.resolved).catch(() => null);
  if (!stat?.isDirectory()) {
    throw new Error(`Working directory does not exist: ${requested}`);
  }
  return resolved.resolved;
}

async function resolveGitRoot(cwd: string): Promise<string | null> {
  try {
    const { stdout } = await runExec("git", ["-C", cwd, "rev-parse", "--show-toplevel"], {
      timeoutMs: 5_000,
      maxBuffer: 64_000,
    });
    const root = stdout.trim();
    return root || null;
  } catch {
    return null;
  }
}

function parseGitStatusPorcelain(raw: string): GitStatusEntry[] {
  if (!raw) {
    return [];
  }
  const tokens = raw.split("\0").filter(Boolean);
  const entries: GitStatusEntry[] = [];
  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i] ?? "";
    if (token.length < 4) {
      continue;
    }
    const code = token.slice(0, 2);
    let filePath = token.slice(3);
    if (
      (code.startsWith("R") || code.endsWith("R") || code.startsWith("C") || code.endsWith("C")) &&
      tokens[i + 1]
    ) {
      filePath = tokens[i + 1] ?? filePath;
      i += 1;
    }
    if (!filePath) {
      continue;
    }
    entries.push({ code, path: filePath });
  }
  return entries;
}

async function readGitStatus(repoRoot: string): Promise<GitStatusEntry[]> {
  const { stdout } = await runExec(
    "git",
    ["-C", repoRoot, "status", "--porcelain=v1", "-z", "--untracked-files=all"],
    { timeoutMs: 10_000, maxBuffer: 512_000 },
  );
  return parseGitStatusPorcelain(stdout);
}

async function createGitBaseline(repoRoot: string): Promise<GitBaseline | null> {
  const statusEntries = await readGitStatus(repoRoot);
  if (statusEntries.length === 0) {
    return null;
  }
  const snapshotRoot = await fs.mkdtemp(path.join(os.tmpdir(), "marv-external-cli-baseline-"));
  const emptyFilePath = path.join(snapshotRoot, "__empty__");
  await fs.writeFile(emptyFilePath, "");
  const entries = new Map<string, BaselineSnapshotEntry>();
  let index = 0;
  for (const entry of statusEntries) {
    const relativePath = entry.path;
    if (entries.has(relativePath)) {
      continue;
    }
    const snapshotPath = path.join(snapshotRoot, `snapshot-${index++}`);
    const absolutePath = path.join(repoRoot, ...relativePath.split("/"));
    const existing = await fs.readFile(absolutePath).catch(() => null);
    await fs.writeFile(snapshotPath, existing ?? "");
    entries.set(relativePath, { relativePath, snapshotPath });
  }
  return { repoRoot, snapshotRoot, emptyFilePath, entries };
}

async function cleanupGitBaseline(baseline: GitBaseline | null): Promise<void> {
  if (!baseline) {
    return;
  }
  await fs.rm(baseline.snapshotRoot, { recursive: true, force: true }).catch(() => {});
}

async function runGitCommand(params: {
  argv: string[];
  cwd: string;
  timeoutMs: number;
}): Promise<{ stdout: string; code: number | null }> {
  const result = await runCommandWithTimeout(params.argv, {
    cwd: params.cwd,
    timeoutMs: params.timeoutMs,
  });
  return {
    stdout: result.stdout,
    code: result.code,
  };
}

function sanitizeNoIndexDiff(params: {
  diff: string;
  beforePath: string;
  afterPath: string;
  relativePath: string;
  beforeExists: boolean;
  afterExists: boolean;
}): string {
  let out = params.diff
    .split(params.beforePath)
    .join(params.beforeExists ? `a/${params.relativePath}` : "/dev/null")
    .split(params.afterPath)
    .join(params.afterExists ? `b/${params.relativePath}` : "/dev/null");
  out = out.replace(
    /^diff --git .+$/m,
    `diff --git a/${params.relativePath} b/${params.relativePath}`,
  );
  out = out.replace(
    /^--- .+$/m,
    `--- ${params.beforeExists ? `a/${params.relativePath}` : "/dev/null"}`,
  );
  out = out.replace(
    /^\+\+\+ .+$/m,
    `+++ ${params.afterExists ? `b/${params.relativePath}` : "/dev/null"}`,
  );
  return out.trim();
}

async function diffSnapshotAgainstCurrent(params: {
  baseline: GitBaseline;
  relativePath: string;
  timeoutMs: number;
}): Promise<string> {
  const snapshot = params.baseline.entries.get(params.relativePath);
  if (!snapshot) {
    return "";
  }
  const currentPath = path.join(params.baseline.repoRoot, ...params.relativePath.split("/"));
  const currentExists = !!(await fs.stat(currentPath).catch(() => null));
  const afterPath = currentExists ? currentPath : params.baseline.emptyFilePath;
  const result = await runGitCommand({
    argv: ["git", "diff", "--no-index", "--", snapshot.snapshotPath, afterPath],
    cwd: params.baseline.repoRoot,
    timeoutMs: params.timeoutMs,
  });
  if (!result.stdout.trim()) {
    return "";
  }
  return sanitizeNoIndexDiff({
    diff: result.stdout,
    beforePath: snapshot.snapshotPath,
    afterPath,
    relativePath: params.relativePath,
    beforeExists: true,
    afterExists: currentExists,
  });
}

async function diffNewUntrackedFile(params: {
  repoRoot: string;
  emptyFilePath: string;
  relativePath: string;
  timeoutMs: number;
}): Promise<string> {
  const currentPath = path.join(params.repoRoot, ...params.relativePath.split("/"));
  const result = await runGitCommand({
    argv: ["git", "diff", "--no-index", "--", params.emptyFilePath, currentPath],
    cwd: params.repoRoot,
    timeoutMs: params.timeoutMs,
  });
  if (!result.stdout.trim()) {
    return "";
  }
  return sanitizeNoIndexDiff({
    diff: result.stdout,
    beforePath: params.emptyFilePath,
    afterPath: currentPath,
    relativePath: params.relativePath,
    beforeExists: false,
    afterExists: true,
  });
}

async function collectIncrementalGitDiff(params: {
  baseline: GitBaseline;
  timeoutMs: number;
}): Promise<GitDiffSummary> {
  const postEntries = new Map(
    (await readGitStatus(params.baseline.repoRoot)).map((entry) => [entry.path, entry]),
  );
  const candidates = new Set<string>([...params.baseline.entries.keys(), ...postEntries.keys()]);
  const filesChanged: string[] = [];
  const diffs: string[] = [];

  for (const relativePath of Array.from(candidates).toSorted()) {
    const hadSnapshot = params.baseline.entries.has(relativePath);
    let diff = "";
    if (hadSnapshot) {
      diff = await diffSnapshotAgainstCurrent({
        baseline: params.baseline,
        relativePath,
        timeoutMs: params.timeoutMs,
      });
    } else {
      const entry = postEntries.get(relativePath);
      if (!entry) {
        continue;
      }
      if (entry.code === "??") {
        diff = await diffNewUntrackedFile({
          repoRoot: params.baseline.repoRoot,
          emptyFilePath: params.baseline.emptyFilePath,
          relativePath,
          timeoutMs: params.timeoutMs,
        });
      } else {
        const result = await runGitCommand({
          argv: [
            "git",
            "-C",
            params.baseline.repoRoot,
            "diff",
            "--no-ext-diff",
            "--",
            relativePath,
          ],
          cwd: params.baseline.repoRoot,
          timeoutMs: params.timeoutMs,
        });
        diff = result.stdout.trim();
      }
    }
    if (!diff) {
      continue;
    }
    filesChanged.push(relativePath);
    diffs.push(diff);
  }

  return {
    filesChanged,
    gitDiff: diffs.length > 0 ? truncate(diffs.join("\n\n"), GIT_DIFF_LIMIT) : undefined,
  };
}

async function collectWorktreeGitDiff(params: {
  repoRoot: string;
  timeoutMs: number;
}): Promise<GitDiffSummary> {
  const names = await runGitCommand({
    argv: ["git", "-C", params.repoRoot, "diff", "--name-only"],
    cwd: params.repoRoot,
    timeoutMs: params.timeoutMs,
  });
  const diff = await runGitCommand({
    argv: ["git", "-C", params.repoRoot, "diff", "--no-ext-diff"],
    cwd: params.repoRoot,
    timeoutMs: params.timeoutMs,
  });
  return {
    filesChanged: names.stdout
      .split(/\r?\n/g)
      .map((line) => line.trim())
      .filter(Boolean),
    gitDiff: truncate(diff.stdout, GIT_DIFF_LIMIT),
  };
}

async function createDetachedWorktree(params: {
  cwd: string;
  timeoutMs: number;
}): Promise<{ worktreePath: string; executionDir: string } | null> {
  const gitRoot = await resolveGitRoot(params.cwd);
  if (!gitRoot) {
    return null;
  }
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "marv-external-cli-worktree-"));
  const worktreePath = path.join(tempRoot, "worktree");
  const addResult = await runCommandWithTimeout(
    ["git", "-C", gitRoot, "worktree", "add", "--detach", worktreePath, "HEAD"],
    {
      cwd: gitRoot,
      timeoutMs: params.timeoutMs,
    },
  );
  if (addResult.code !== 0) {
    await fs.rm(tempRoot, { recursive: true, force: true }).catch(() => {});
    throw new Error(addResult.stderr.trim() || "Failed to create detached worktree.");
  }
  const relative = path.relative(gitRoot, params.cwd);
  const executionDir =
    !relative || relative === "" ? worktreePath : path.join(worktreePath, relative);
  return { worktreePath, executionDir };
}

function readBooleanParam(params: Record<string, unknown>, key: string): boolean | undefined {
  const raw = params[key];
  return typeof raw === "boolean" ? raw : undefined;
}

function resolveTargetCli(params: {
  explicitCli: ExternalCliAdapterId | null;
  cfg: ExternalCliToolsConfig | undefined;
}): { type: "configured"; cli: ExternalCliAdapterId } | { type: "not_configured" } {
  if (params.explicitCli) {
    return { type: "configured", cli: params.explicitCli };
  }
  const defaultCli = params.cfg?.defaultCli;
  if (defaultCli) {
    return { type: "configured", cli: defaultCli };
  }
  const available = normalizeAvailableCli(params.cfg?.availableCli);
  if (available.length > 0) {
    return { type: "configured", cli: available[0] };
  }
  return { type: "not_configured" };
}

function buildNotConfiguredResult(): ReturnType<typeof jsonResult> {
  return jsonResult({
    status: "not_configured" satisfies ExternalCliResultStatus,
    error:
      "External CLI is enabled, but no available CLI brands are configured. Ask the user which external CLI brands are installed on this machine, then store them with self_settings.",
  });
}

export function createExternalCliTool(options?: {
  config?: MarvConfig;
  workspaceDir?: string;
  sandboxed?: boolean;
}): AnyAgentTool | null {
  if (!isExternalCliEnabled(options?.config) || options?.sandboxed === true) {
    return null;
  }
  const workspaceRoot = resolveWorkspaceRoot(options?.workspaceDir);
  return {
    label: "External CLI",
    name: "external_cli",
    description:
      "Delegate a difficult task to a stronger local external AI CLI such as Codex, Claude Code, Aider, or Gemini. Use this as an explicit fallback when the task is complex, your current approach is weak, or your result quality is below the expected bar. If the user's available external CLI brands are not configured yet, ask which brands are installed and store them via self_settings before retrying.",
    parameters: ExternalCliToolSchema,
    execute: async (_toolCallId, args) => {
      const params = args as Record<string, unknown>;
      const task = readStringParam(params, "task", { required: true });
      const explicitCli = normalizeExternalCliId(readStringParam(params, "cli"));
      const target = resolveTargetCli({
        explicitCli,
        cfg: resolveExternalCliConfig(options?.config),
      });
      if (target.type === "not_configured") {
        return buildNotConfiguredResult();
      }

      const kind = readStringParam(params, "kind") ?? "general";
      const timeoutSeconds =
        readNumberParam(params, "timeoutSeconds") ??
        resolveExternalCliConfig(options?.config)?.timeoutSeconds ??
        DEFAULT_TIMEOUT_SECONDS;
      const timeoutMs = Math.max(10, Math.trunc(timeoutSeconds)) * 1_000;
      const requestedWorkdir = readStringParam(params, "workdir");
      const workdir = await resolveWorkdir({
        workspaceRoot,
        requested: requestedWorkdir,
      });
      const isolate = readBooleanParam(params, "isolate") === true;
      const captureGitDiffParam = readBooleanParam(params, "captureGitDiff");
      const captureGitDiff =
        captureGitDiffParam ??
        resolveExternalCliConfig(options?.config)?.captureGitDiffDefault ??
        true;
      const adapter = getExternalCliAdapter(target.cli);
      const override = resolveExternalCliConfig(options?.config)?.overrides?.[target.cli];
      const command = override?.command?.trim() || adapter.command;
      if (!(await adapter.detect(command))) {
        return jsonResult({
          status: "not_available" satisfies ExternalCliResultStatus,
          cli: target.cli,
          error: `External CLI '${target.cli}' is not installed or not available on PATH.`,
        });
      }

      const model =
        readStringParam(params, "model") ??
        (override?.model?.trim() ? override.model.trim() : undefined);
      const invocation = adapter.buildInvocation({
        task,
        model,
        override,
      });

      let baseline: GitBaseline | null = null;
      let worktreePath: string | undefined;
      let executionDir = workdir;
      if (isolate) {
        const isolated = await createDetachedWorktree({ cwd: workdir, timeoutMs });
        if (!isolated) {
          return jsonResult({
            status: "error" satisfies ExternalCliResultStatus,
            cli: target.cli,
            error: "Detached worktree isolation requires a git repository.",
          });
        }
        worktreePath = isolated.worktreePath;
        executionDir = isolated.executionDir;
      } else if (kind === "coding" && captureGitDiff) {
        const gitRoot = await resolveGitRoot(workdir);
        if (gitRoot) {
          baseline = await createGitBaseline(gitRoot);
        }
      }

      const startedAt = Date.now();
      try {
        const run = await runCommandWithTimeout([invocation.command, ...invocation.args], {
          cwd: executionDir,
          timeoutMs,
          input: invocation.input,
          env: override?.env,
        });
        const parsed = adapter.parseOutput(run.stdout, run.stderr, run.code);
        const combined = [parsed.text, parsed.raw, run.stderr].filter(Boolean).join("\n");
        const status: ExternalCliResultStatus =
          run.termination === "timeout" || run.termination === "no-output-timeout"
            ? "timeout"
            : detectQuotaExhausted(combined)
              ? "quota_exhausted"
              : run.code === 0
                ? "ok"
                : "error";

        let gitSummary: GitDiffSummary | undefined;
        if (kind === "coding" && captureGitDiff) {
          if (worktreePath) {
            gitSummary = await collectWorktreeGitDiff({ repoRoot: worktreePath, timeoutMs });
          } else if (baseline) {
            gitSummary = await collectIncrementalGitDiff({ baseline, timeoutMs });
          }
        }

        return jsonResult({
          status,
          cli: target.cli,
          exitCode: run.code,
          durationMs: Date.now() - startedAt,
          output: truncate(parsed.text || parsed.raw || run.stdout, OUTPUT_LIMIT) ?? "",
          stderr: truncate(run.stderr, STDERR_LIMIT),
          filesChanged: gitSummary?.filesChanged,
          gitDiff: gitSummary?.gitDiff,
          workdir: executionDir,
          worktreePath,
          exitReason:
            status === "quota_exhausted"
              ? "quota_exhausted"
              : run.termination === "timeout" || run.termination === "no-output-timeout"
                ? run.termination
                : undefined,
        });
      } finally {
        await cleanupGitBaseline(baseline);
      }
    },
  };
}

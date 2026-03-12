import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { runCommandWithTimeout, runExec } from "../../process/exec.js";
import { assertSandboxPath } from "../sandbox/sandbox-paths.js";
import type { ManagedCliManifest, ManagedCliOutputMode } from "./cli-profile-types.js";

const OUTPUT_LIMIT = 32_000;
const STDERR_LIMIT = 8_000;
const GIT_DIFF_LIMIT = 16_000;
const DEFAULT_TIMEOUT_SECONDS = 300;

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

export type ManagedCliExecutionResult = {
  status: "ok" | "error" | "timeout";
  exitCode: number | null;
  durationMs: number;
  output: string;
  stderr?: string;
  parsed?: unknown;
  filesChanged?: string[];
  gitDiff?: string;
  workdir: string;
  worktreePath?: string;
  exitReason?: string;
};

export type ManagedCliVerificationResult = {
  ok: boolean;
  message: string;
  helpExitCode?: number | null;
  smokeExitCode?: number | null;
};

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

function parseOutput(stdout: string, outputMode: ManagedCliOutputMode): unknown {
  const trimmed = stdout.trim();
  if (!trimmed) {
    return undefined;
  }
  if (outputMode === "text") {
    return trimmed;
  }
  if (outputMode === "json") {
    return JSON.parse(trimmed);
  }
  const lines = trimmed
    .split(/\r?\n/g)
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.map((line) => JSON.parse(line));
}

function normalizeTemplateVars(raw?: string): Record<string, string> {
  if (!raw?.trim()) {
    return {};
  }
  const parsed = JSON.parse(raw) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("templateVarsJson must be a JSON object.");
  }
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(parsed)) {
    if (value === null || value === undefined) {
      continue;
    }
    out[key] = typeof value === "string" ? value : JSON.stringify(value);
  }
  return out;
}

function renderTemplate(template: string, vars: Record<string, string>): string {
  return template.replace(/\{\{\s*([a-zA-Z0-9_.-]+)\s*\}\}/g, (_match, key: string) => {
    if (!(key in vars)) {
      throw new Error(`Missing template variable: ${key}`);
    }
    return vars[key] ?? "";
  });
}

function resolveCommandPath(toolDir: string, rawCommand: string): string {
  const trimmed = rawCommand.trim();
  if (!trimmed) {
    throw new Error("CLI command is required.");
  }
  if (path.isAbsolute(trimmed)) {
    return trimmed;
  }
  if (trimmed.startsWith("./") || trimmed.startsWith("../") || trimmed.includes(path.sep)) {
    return path.resolve(toolDir, trimmed);
  }
  return trimmed;
}

function resolveStaticArg(toolDir: string, value: string): string {
  if (path.isAbsolute(value)) {
    return value;
  }
  if (value.startsWith("./") || value.startsWith("../")) {
    return path.resolve(toolDir, value);
  }
  return value;
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
    return stdout.trim() || null;
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
  const snapshotRoot = await fs.mkdtemp(path.join(os.tmpdir(), "marv-cli-profile-baseline-"));
  const emptyFilePath = path.join(snapshotRoot, "__empty__");
  await fs.writeFile(emptyFilePath, "");
  const entries = new Map<string, BaselineSnapshotEntry>();
  let index = 0;
  for (const entry of statusEntries) {
    if (entries.has(entry.path)) {
      continue;
    }
    const snapshotPath = path.join(snapshotRoot, `snapshot-${index++}`);
    const absolutePath = path.join(repoRoot, ...entry.path.split("/"));
    const existing = await fs.readFile(absolutePath).catch(() => null);
    await fs.writeFile(snapshotPath, existing ?? "");
    entries.set(entry.path, { relativePath: entry.path, snapshotPath });
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
  return { stdout: result.stdout, code: result.code };
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
}): Promise<{ repoRoot: string; worktreePath: string; executionDir: string } | null> {
  const gitRoot = await resolveGitRoot(params.cwd);
  if (!gitRoot) {
    return null;
  }
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "marv-cli-profile-worktree-"));
  const worktreePath = path.join(tempRoot, "worktree");
  const addResult = await runCommandWithTimeout(
    ["git", "-C", gitRoot, "worktree", "add", "--detach", worktreePath, "HEAD"],
    { cwd: gitRoot, timeoutMs: params.timeoutMs },
  );
  if (addResult.code !== 0) {
    await fs.rm(tempRoot, { recursive: true, force: true }).catch(() => {});
    throw new Error(addResult.stderr.trim() || "Failed to create detached worktree.");
  }
  const relative = path.relative(gitRoot, params.cwd);
  const executionDir =
    !relative || relative === "" ? worktreePath : path.join(worktreePath, relative);
  return { repoRoot: gitRoot, worktreePath, executionDir };
}

async function cleanupDetachedWorktree(params: {
  repoRoot: string;
  worktreePath?: string;
}): Promise<void> {
  if (!params.worktreePath) {
    return;
  }
  await runCommandWithTimeout(
    ["git", "-C", params.repoRoot, "worktree", "remove", "--force", params.worktreePath],
    { cwd: params.repoRoot, timeoutMs: 10_000 },
  ).catch(() => {});
  await fs.rm(path.dirname(params.worktreePath), { recursive: true, force: true }).catch(() => {});
}

function formatExecutionError(error: unknown): string {
  if (error instanceof Error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      return `Command not found: ${error.message}`;
    }
    return error.message;
  }
  return String(error);
}

function describeVerificationFailure(params: {
  phase: "help" | "smoke";
  result: ManagedCliExecutionResult;
}): string {
  if (params.result.status === "timeout") {
    return `CLI ${params.phase} command timed out during verification.`;
  }
  const detail = params.result.stderr || params.result.output;
  if (detail) {
    return `CLI ${params.phase} command failed verification: ${detail}`;
  }
  if (params.result.exitCode !== null) {
    return `CLI ${params.phase} command failed verification with exit code ${params.result.exitCode}.`;
  }
  return `CLI ${params.phase} command failed verification.`;
}

export function buildManagedCliArgv(params: {
  manifest: ManagedCliManifest;
  templateVarsJson?: string;
  extraArgs?: string[];
}): string[] {
  const vars = normalizeTemplateVars(params.templateVarsJson);
  const command = resolveCommandPath(params.manifest.toolDir, params.manifest.entry.command);
  const staticArgs = (params.manifest.entry.staticArgs ?? []).map((value) =>
    resolveStaticArg(params.manifest.toolDir, value),
  );
  const templatedArgs = (params.manifest.entry.argsTemplate ?? []).map((value) =>
    renderTemplate(resolveStaticArg(params.manifest.toolDir, value), vars),
  );
  return [command, ...staticArgs, ...templatedArgs, ...(params.extraArgs ?? [])];
}

export async function executeManagedCliProfile(params: {
  manifest: ManagedCliManifest;
  workspaceRoot: string;
  workdir?: string;
  templateVarsJson?: string;
  extraArgs?: string[];
  timeoutSeconds?: number;
  input?: string;
  captureGitDiff?: boolean;
  isolate?: boolean;
}): Promise<ManagedCliExecutionResult> {
  const timeoutSeconds = Math.max(10, Math.trunc(params.timeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS));
  const timeoutMs = timeoutSeconds * 1_000;
  const argv = buildManagedCliArgv({
    manifest: params.manifest,
    templateVarsJson: params.templateVarsJson,
    extraArgs: params.extraArgs,
  });
  const requestedWorkdir = await resolveWorkdir({
    workspaceRoot: params.workspaceRoot,
    requested: params.workdir,
  });
  let baseline: GitBaseline | null = null;
  let isolatedRepoRoot: string | undefined;
  let worktreePath: string | undefined;
  let executionDir = requestedWorkdir;
  if (params.isolate) {
    const isolated = await createDetachedWorktree({ cwd: requestedWorkdir, timeoutMs });
    if (!isolated) {
      throw new Error("Detached worktree isolation requires a git repository.");
    }
    isolatedRepoRoot = isolated.repoRoot;
    worktreePath = isolated.worktreePath;
    executionDir = isolated.executionDir;
  } else if (params.captureGitDiff === true) {
    const gitRoot = await resolveGitRoot(requestedWorkdir);
    if (gitRoot) {
      baseline = await createGitBaseline(gitRoot);
    }
  }
  const startedAt = Date.now();
  try {
    let run;
    try {
      run = await runCommandWithTimeout(argv, {
        cwd: executionDir,
        timeoutMs,
        input: params.input,
        env: params.manifest.entry.env,
      });
    } catch (error) {
      return {
        status: "error",
        exitCode: null,
        durationMs: Date.now() - startedAt,
        output: "",
        stderr: truncate(formatExecutionError(error), STDERR_LIMIT),
        workdir: executionDir,
        worktreePath,
      };
    }
    let parsed: unknown;
    try {
      parsed = parseOutput(run.stdout, params.manifest.entry.outputMode);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        status: "error",
        exitCode: run.code,
        durationMs: Date.now() - startedAt,
        output: truncate(run.stdout, OUTPUT_LIMIT) ?? "",
        stderr: truncate(`${run.stderr}\n${message}`.trim(), STDERR_LIMIT),
        workdir: executionDir,
        worktreePath,
      };
    }
    let gitSummary: GitDiffSummary | undefined;
    if (params.captureGitDiff === true) {
      if (worktreePath) {
        gitSummary = await collectWorktreeGitDiff({ repoRoot: worktreePath, timeoutMs });
      } else if (baseline) {
        gitSummary = await collectIncrementalGitDiff({ baseline, timeoutMs });
      }
    }
    return {
      status:
        run.termination === "timeout" || run.termination === "no-output-timeout"
          ? "timeout"
          : run.code === 0
            ? "ok"
            : "error",
      exitCode: run.code,
      durationMs: Date.now() - startedAt,
      output: truncate(run.stdout, OUTPUT_LIMIT) ?? "",
      stderr: truncate(run.stderr, STDERR_LIMIT),
      parsed,
      filesChanged: gitSummary?.filesChanged,
      gitDiff: gitSummary?.gitDiff,
      workdir: executionDir,
      worktreePath,
      exitReason:
        run.termination === "timeout" || run.termination === "no-output-timeout"
          ? run.termination
          : undefined,
    };
  } finally {
    await cleanupGitBaseline(baseline);
    if (isolatedRepoRoot && worktreePath) {
      await cleanupDetachedWorktree({ repoRoot: isolatedRepoRoot, worktreePath });
    }
  }
}

export async function verifyManagedCliProfile(params: {
  manifest: ManagedCliManifest;
  workspaceRoot: string;
  timeoutSeconds?: number;
}): Promise<ManagedCliVerificationResult> {
  const helpArgs = params.manifest.verification?.helpArgs;
  const smokeArgs = params.manifest.verification?.smokeArgs;
  const helpCheckEnabled = !!(helpArgs && helpArgs.length > 0);
  if (helpCheckEnabled) {
    const helpRun = await executeManagedCliProfile({
      manifest: params.manifest,
      workspaceRoot: params.workspaceRoot,
      extraArgs: helpArgs,
      timeoutSeconds: params.timeoutSeconds,
    });
    if (helpRun.exitCode !== 0) {
      return {
        ok: false,
        helpExitCode: helpRun.exitCode,
        message: describeVerificationFailure({ phase: "help", result: helpRun }),
      };
    }
  }
  const smokeRun = await executeManagedCliProfile({
    manifest: params.manifest,
    workspaceRoot: params.workspaceRoot,
    extraArgs: smokeArgs && smokeArgs.length > 0 ? smokeArgs : undefined,
    timeoutSeconds: params.timeoutSeconds,
  });
  if (smokeRun.exitCode !== 0) {
    return {
      ok: false,
      helpExitCode: helpCheckEnabled ? 0 : undefined,
      smokeExitCode: smokeRun.exitCode,
      message: describeVerificationFailure({ phase: "smoke", result: smokeRun }),
    };
  }
  return {
    ok: true,
    helpExitCode: helpCheckEnabled ? 0 : undefined,
    smokeExitCode: smokeRun.exitCode,
    message: "CLI profile verified successfully.",
  };
}

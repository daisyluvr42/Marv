import { type CommandOptions } from "../../process/exec.js";
import { trimLogTail } from "../restart-sentinel.js";

// ── Types ─────────────────────────────────────────────────────────────

export type CommandRunner = (
  argv: string[],
  options: CommandOptions,
) => Promise<{ stdout: string; stderr: string; code: number | null }>;

export type UpdateStepResult = {
  name: string;
  command: string;
  cwd: string;
  durationMs: number;
  exitCode: number | null;
  stdoutTail?: string | null;
  stderrTail?: string | null;
};

export type UpdateStepInfo = {
  name: string;
  command: string;
  index: number;
  total: number;
};

export type UpdateStepCompletion = UpdateStepInfo & {
  durationMs: number;
  exitCode: number | null;
  stderrTail?: string | null;
};

export type UpdateStepProgress = {
  onStepStart?: (step: UpdateStepInfo) => void;
  onStepComplete?: (step: UpdateStepCompletion) => void;
};

export type RunStepOptions = {
  runCommand: CommandRunner;
  name: string;
  argv: string[];
  cwd: string;
  timeoutMs: number;
  env?: NodeJS.ProcessEnv;
  progress?: UpdateStepProgress;
  stepIndex: number;
  totalSteps: number;
};

// ── Constants ─────────────────────────────────────────────────────────

export const MAX_LOG_CHARS = 8000;

// ── Step runner ───────────────────────────────────────────────────────

export async function runStep(opts: RunStepOptions): Promise<UpdateStepResult> {
  const { runCommand, name, argv, cwd, timeoutMs, env, progress, stepIndex, totalSteps } = opts;
  const command = argv.join(" ");

  const stepInfo: UpdateStepInfo = {
    name,
    command,
    index: stepIndex,
    total: totalSteps,
  };

  progress?.onStepStart?.(stepInfo);

  const started = Date.now();
  const result = await runCommand(argv, { cwd, timeoutMs, env });
  const durationMs = Date.now() - started;

  const stderrTail = trimLogTail(result.stderr, MAX_LOG_CHARS);

  progress?.onStepComplete?.({
    ...stepInfo,
    durationMs,
    exitCode: result.code,
    stderrTail,
  });

  return {
    name,
    command,
    cwd,
    durationMs,
    exitCode: result.code,
    stdoutTail: trimLogTail(result.stdout, MAX_LOG_CHARS),
    stderrTail: trimLogTail(result.stderr, MAX_LOG_CHARS),
  };
}

// ── Package manager helpers ───────────────────────────────────────────

export type PackageManager = "pnpm" | "bun" | "npm";

export function managerScriptArgs(
  manager: PackageManager,
  script: string,
  args: string[] = [],
): string[] {
  if (manager === "pnpm") {
    return ["pnpm", script, ...args];
  }
  if (manager === "bun") {
    return ["bun", "run", script, ...args];
  }
  if (args.length > 0) {
    return ["npm", "run", script, "--", ...args];
  }
  return ["npm", "run", script];
}

export function managerInstallArgs(manager: PackageManager): string[] {
  if (manager === "pnpm") {
    return ["pnpm", "install"];
  }
  if (manager === "bun") {
    return ["bun", "install"];
  }
  return ["npm", "install"];
}

// ── Git helpers ───────────────────────────────────────────────────────

export async function getGitHeadSha(
  runCommand: CommandRunner,
  root: string,
  timeoutMs: number,
): Promise<string | null> {
  const res = await runCommand(["git", "-C", root, "rev-parse", "HEAD"], {
    cwd: root,
    timeoutMs,
  }).catch(() => null);
  if (!res || res.code !== 0) {
    return null;
  }
  const sha = res.stdout.trim();
  return sha || null;
}

// ── Rollback ──────────────────────────────────────────────────────────

export async function rollbackGitCheckout(params: {
  runCommand: CommandRunner;
  root: string;
  timeoutMs: number;
  manager: PackageManager;
  progress?: UpdateStepProgress;
  steps: UpdateStepResult[];
  targetSha: string;
  totalSteps: number;
  startIndex: number;
}): Promise<{ ok: boolean; rolledBackToSha: string | null }> {
  let stepIndex = params.startIndex;
  const nextStep = (name: string, argv: string[]): RunStepOptions => ({
    runCommand: params.runCommand,
    name,
    argv,
    cwd: params.root,
    timeoutMs: params.timeoutMs,
    progress: params.progress,
    stepIndex: stepIndex++,
    totalSteps: params.totalSteps,
  });

  const resetStep = await runStep(
    nextStep("rollback git reset --hard", [
      "git",
      "-C",
      params.root,
      "reset",
      "--hard",
      params.targetSha,
    ]),
  );
  params.steps.push(resetStep);
  if (resetStep.exitCode !== 0) {
    return { ok: false, rolledBackToSha: null };
  }

  const depsStep = await runStep(
    nextStep("rollback deps install", managerInstallArgs(params.manager)),
  );
  params.steps.push(depsStep);
  if (depsStep.exitCode !== 0) {
    return { ok: false, rolledBackToSha: params.targetSha };
  }

  const buildStep = await runStep(
    nextStep("rollback build", managerScriptArgs(params.manager, "build")),
  );
  params.steps.push(buildStep);
  if (buildStep.exitCode !== 0) {
    return { ok: false, rolledBackToSha: params.targetSha };
  }

  const uiBuildStep = await runStep(
    nextStep("rollback ui:build", managerScriptArgs(params.manager, "ui:build")),
  );
  params.steps.push(uiBuildStep);
  if (uiBuildStep.exitCode !== 0) {
    return { ok: false, rolledBackToSha: params.targetSha };
  }

  const doctorEntry = (await import("node:path")).join(params.root, "marv.mjs");
  const doctorStep = await runStep(
    nextStep("rollback marv doctor", [
      process.execPath,
      doctorEntry,
      "doctor",
      "--non-interactive",
      "--fix",
    ]),
  );
  params.steps.push(doctorStep);
  if (doctorStep.exitCode !== 0) {
    return { ok: false, rolledBackToSha: params.targetSha };
  }

  return { ok: true, rolledBackToSha: params.targetSha };
}

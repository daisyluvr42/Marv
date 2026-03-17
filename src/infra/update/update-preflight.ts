import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { checkCiStatus } from "./ci-status.js";
import {
  type CommandRunner,
  managerInstallArgs,
  managerScriptArgs,
  type PackageManager,
  runStep,
  type UpdateStepProgress,
  type UpdateStepResult,
} from "./update-steps.js";

export const PREFLIGHT_MAX_COMMITS = 10;

// ── Types ─────────────────────────────────────────────────────────────

export type PreflightParams = {
  runCommand: CommandRunner;
  gitRoot: string;
  manager: PackageManager;
  timeoutMs: number;
  progress?: UpdateStepProgress;
  /** The base SHA to create the worktree from (usually upstream HEAD). */
  baseSha: string;
  /** Candidate SHAs to test, newest first. */
  candidates: string[];
  /** Starting step index for progress tracking. */
  stepIndexStart: number;
  /** Total steps for progress tracking. */
  totalSteps: number;
  /** If true, trust CI-passed commits and skip local preflight. Defaults to true. */
  trustCi?: boolean;
};

export type PreflightResult = {
  selectedSha: string | null;
  steps: UpdateStepResult[];
  stepCount: number;
};

// ── Preflight runner ──────────────────────────────────────────────────

/**
 * Run preflight validation in a temporary git worktree.
 *
 * For each candidate SHA, checks out the commit, installs deps, builds,
 * and lints. Returns the first candidate that passes all checks.
 */
export async function runPreflight(params: PreflightParams): Promise<PreflightResult> {
  const { runCommand, gitRoot, manager, timeoutMs, progress, baseSha, candidates } = params;
  const trustCi = params.trustCi ?? true;
  const steps: UpdateStepResult[] = [];
  let stepIndex = params.stepIndexStart;

  const step = (name: string, argv: string[], cwd: string) => ({
    runCommand,
    name,
    argv,
    cwd,
    timeoutMs,
    progress,
    stepIndex: stepIndex++,
    totalSteps: params.totalSteps,
  });

  // CI fast path: if the newest candidate passed CI, skip the full worktree preflight
  if (trustCi && candidates.length > 0) {
    for (const sha of candidates) {
      const ciResult = await checkCiStatus({ runCommand, root: gitRoot, sha, timeoutMs });
      if (ciResult.passed && ciResult.source !== "unavailable") {
        return { selectedSha: sha, steps, stepCount: 0 };
      }
      // Only check the first candidate via CI; if it didn't pass, fall through
      // to the full preflight for all candidates.
      break;
    }
  }

  const preflightRoot = await fs.mkdtemp(path.join(os.tmpdir(), "marv-update-preflight-"));
  const worktreeDir = path.join(preflightRoot, "worktree");

  const worktreeStep = await runStep(
    step(
      "preflight worktree",
      ["git", "-C", gitRoot, "worktree", "add", "--detach", worktreeDir, baseSha],
      gitRoot,
    ),
  );
  steps.push(worktreeStep);

  if (worktreeStep.exitCode !== 0) {
    await fs.rm(preflightRoot, { recursive: true, force: true }).catch(() => {});
    return { selectedSha: null, steps, stepCount: stepIndex - params.stepIndexStart };
  }

  let selectedSha: string | null = null;
  try {
    for (const sha of candidates) {
      const shortSha = sha.slice(0, 8);

      const checkoutStep = await runStep(
        step(
          `preflight checkout (${shortSha})`,
          ["git", "-C", worktreeDir, "checkout", "--detach", sha],
          worktreeDir,
        ),
      );
      steps.push(checkoutStep);
      if (checkoutStep.exitCode !== 0) {
        continue;
      }

      const depsStep = await runStep(
        step(`preflight deps install (${shortSha})`, managerInstallArgs(manager), worktreeDir),
      );
      steps.push(depsStep);
      if (depsStep.exitCode !== 0) {
        continue;
      }

      const buildStep = await runStep(
        step(`preflight build (${shortSha})`, managerScriptArgs(manager, "build"), worktreeDir),
      );
      steps.push(buildStep);
      if (buildStep.exitCode !== 0) {
        continue;
      }

      const lintStep = await runStep(
        step(`preflight lint (${shortSha})`, managerScriptArgs(manager, "lint"), worktreeDir),
      );
      steps.push(lintStep);
      if (lintStep.exitCode !== 0) {
        continue;
      }

      selectedSha = sha;
      break;
    }
  } finally {
    const removeStep = await runStep(
      step(
        "preflight cleanup",
        ["git", "-C", gitRoot, "worktree", "remove", "--force", worktreeDir],
        gitRoot,
      ),
    );
    steps.push(removeStep);
    await runCommand(["git", "-C", gitRoot, "worktree", "prune"], {
      cwd: gitRoot,
      timeoutMs,
    }).catch(() => null);
    await fs.rm(preflightRoot, { recursive: true, force: true }).catch(() => {});
  }

  return { selectedSha, steps, stepCount: stepIndex - params.stepIndexStart };
}

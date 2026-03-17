import fs from "node:fs/promises";
import path from "node:path";
import { runCommandWithTimeout } from "../../process/exec.js";
import {
  resolveControlUiDistIndexHealth,
  resolveControlUiDistIndexPathForRoot,
} from "../control-ui-assets.js";
import { detectPackageManager as detectPackageManagerImpl } from "../detect-package-manager.js";
import { readPackageName, readPackageVersion } from "../package-json.js";
import { trimLogTail } from "../restart-sentinel.js";
import {
  normalizeUpdateApprovalConfig,
  resolveDeployApproval,
  type UpdateApprovalConfig,
} from "./deploy-approval.js";
import {
  beginDeployAttempt,
  finishDeployAttempt,
  readDeployState,
  recordDeployRollback,
  resolveDeployStatePath,
} from "./deploy-state.js";
import {
  channelToNpmTag,
  DEFAULT_PACKAGE_CHANNEL,
  DEV_BRANCH,
  isBetaTag,
  isStableTag,
  type UpdateChannel,
} from "./update-channels.js";
import { compareSemverStrings } from "./update-check.js";
import {
  cleanupGlobalRenameDirs,
  detectGlobalInstallManagerForRoot,
  globalInstallArgs,
} from "./update-global.js";
import { PREFLIGHT_MAX_COMMITS, runPreflight } from "./update-preflight.js";
import {
  type CommandRunner,
  getGitHeadSha,
  managerInstallArgs,
  managerScriptArgs,
  MAX_LOG_CHARS,
  rollbackGitCheckout,
  runStep,
  type RunStepOptions,
  type UpdateStepProgress,
  type UpdateStepResult,
} from "./update-steps.js";

// Re-export types that consumers depend on
export type { UpdateStepResult, UpdateStepProgress, CommandRunner };
export type { UpdateStepInfo, UpdateStepCompletion } from "./update-steps.js";

export type UpdateRunResult = {
  status: "ok" | "error" | "skipped";
  mode: "git" | "pnpm" | "bun" | "npm" | "unknown";
  root?: string;
  reason?: string;
  before?: { sha?: string | null; version?: string | null };
  after?: { sha?: string | null; version?: string | null };
  deploy?: {
    statePath?: string;
    lastKnownGoodSha?: string | null;
    targetSha?: string | null;
    rolledBackToSha?: string | null;
  };
  steps: UpdateStepResult[];
  durationMs: number;
};

type UpdateRunnerOptions = {
  cwd?: string;
  argv1?: string;
  tag?: string;
  channel?: UpdateChannel;
  approval?: UpdateApprovalConfig | null;
  timeoutMs?: number;
  runCommand?: CommandRunner;
  progress?: UpdateStepProgress;
  trustCi?: boolean;
};

const DEFAULT_TIMEOUT_MS = 20 * 60_000;
const START_DIRS = ["cwd", "argv1", "process"];
const DEFAULT_PACKAGE_NAME = "agentmarv";
const CORE_PACKAGE_NAMES = new Set([DEFAULT_PACKAGE_NAME]);

// ── Root detection helpers ────────────────────────────────────────────

function normalizeDir(value?: string | null) {
  if (!value) {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  return path.resolve(trimmed);
}

function resolveNodeModulesBinPackageRoot(argv1: string): string | null {
  const normalized = path.resolve(argv1);
  const parts = normalized.split(path.sep);
  const binIndex = parts.lastIndexOf(".bin");
  if (binIndex <= 0) {
    return null;
  }
  if (parts[binIndex - 1] !== "node_modules") {
    return null;
  }
  const nodeModulesDir = parts.slice(0, binIndex).join(path.sep);
  return path.join(nodeModulesDir, DEFAULT_PACKAGE_NAME);
}

function buildStartDirs(opts: UpdateRunnerOptions): string[] {
  const dirs: string[] = [];
  const cwd = normalizeDir(opts.cwd);
  if (cwd) {
    dirs.push(cwd);
  }
  const argv1 = normalizeDir(opts.argv1);
  if (argv1) {
    dirs.push(path.dirname(argv1));
    const packageRoot = resolveNodeModulesBinPackageRoot(argv1);
    if (packageRoot) {
      dirs.push(packageRoot);
    }
  }
  const proc = normalizeDir(process.cwd());
  if (proc) {
    dirs.push(proc);
  }
  return Array.from(new Set(dirs));
}

// ── Git helpers ───────────────────────────────────────────────────────

async function readBranchName(
  runCommand: CommandRunner,
  root: string,
  timeoutMs: number,
): Promise<string | null> {
  const res = await runCommand(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"], {
    timeoutMs,
  }).catch(() => null);
  if (!res || res.code !== 0) {
    return null;
  }
  const branch = res.stdout.trim();
  return branch || null;
}

async function listGitTags(
  runCommand: CommandRunner,
  root: string,
  timeoutMs: number,
  pattern = "v*",
): Promise<string[]> {
  const res = await runCommand(["git", "-C", root, "tag", "--list", pattern, "--sort=-v:refname"], {
    timeoutMs,
  }).catch(() => null);
  if (!res || res.code !== 0) {
    return [];
  }
  return res.stdout
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

async function resolveChannelTag(
  runCommand: CommandRunner,
  root: string,
  timeoutMs: number,
  channel: Exclude<UpdateChannel, "dev">,
): Promise<string | null> {
  const tags = await listGitTags(runCommand, root, timeoutMs);
  if (channel === "beta") {
    const betaTag = tags.find((tag) => isBetaTag(tag)) ?? null;
    const stableTag = tags.find((tag) => isStableTag(tag)) ?? null;
    if (!betaTag) {
      return stableTag;
    }
    if (!stableTag) {
      return betaTag;
    }
    const cmp = compareSemverStrings(betaTag, stableTag);
    if (cmp != null && cmp < 0) {
      return stableTag;
    }
    return betaTag;
  }
  return tags.find((tag) => isStableTag(tag)) ?? null;
}

async function resolveGitRoot(
  runCommand: CommandRunner,
  candidates: string[],
  timeoutMs: number,
): Promise<string | null> {
  for (const dir of candidates) {
    const res = await runCommand(["git", "-C", dir, "rev-parse", "--show-toplevel"], {
      timeoutMs,
    });
    if (res.code === 0) {
      const root = res.stdout.trim();
      if (root) {
        return root;
      }
    }
  }
  return null;
}

async function findPackageRoot(candidates: string[]) {
  for (const dir of candidates) {
    let current = dir;
    for (let i = 0; i < 12; i += 1) {
      const pkgPath = path.join(current, "package.json");
      try {
        const raw = await fs.readFile(pkgPath, "utf-8");
        const parsed = JSON.parse(raw) as { name?: string };
        const name = parsed?.name?.trim();
        if (name && CORE_PACKAGE_NAMES.has(name)) {
          return current;
        }
      } catch {
        // ignore
      }
      const parent = path.dirname(current);
      if (parent === current) {
        break;
      }
      current = parent;
    }
  }
  return null;
}

async function detectPackageManager(root: string) {
  return (await detectPackageManagerImpl(root)) ?? "npm";
}

async function isCorePackageRoot(root: string): Promise<boolean> {
  const packageName = await readPackageName(root);
  return Boolean(packageName && CORE_PACKAGE_NAMES.has(packageName));
}

function normalizeTag(tag?: string) {
  const trimmed = tag?.trim();
  if (!trimmed) {
    return "latest";
  }
  if (trimmed.startsWith(`${DEFAULT_PACKAGE_NAME}@`)) {
    return trimmed.slice(`${DEFAULT_PACKAGE_NAME}@`.length);
  }
  return trimmed;
}

// ── Main update ───────────────────────────────────────────────────────

export async function runGatewayUpdate(opts: UpdateRunnerOptions = {}): Promise<UpdateRunResult> {
  const startedAt = Date.now();
  const runCommand: CommandRunner =
    opts.runCommand ??
    (async (argv, options) => {
      const res = await runCommandWithTimeout(argv, options);
      return { stdout: res.stdout, stderr: res.stderr, code: res.code };
    });
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const progress = opts.progress;
  const steps: UpdateStepResult[] = [];
  const candidates = buildStartDirs(opts);

  let stepIndex = 0;
  let gitTotalSteps = 0;

  const step = (
    name: string,
    argv: string[],
    cwd: string,
    env?: NodeJS.ProcessEnv,
  ): RunStepOptions => {
    const currentIndex = stepIndex;
    stepIndex += 1;
    return {
      runCommand,
      name,
      argv,
      cwd,
      timeoutMs,
      env,
      progress,
      stepIndex: currentIndex,
      totalSteps: gitTotalSteps,
    };
  };

  const pkgRoot = await findPackageRoot(candidates);

  const detectedGitRoot = await resolveGitRoot(runCommand, candidates, timeoutMs);
  const gitRoot =
    detectedGitRoot && (await isCorePackageRoot(detectedGitRoot)) ? detectedGitRoot : null;

  if (detectedGitRoot && !gitRoot && !pkgRoot) {
    return {
      status: "error",
      mode: "unknown",
      root: detectedGitRoot,
      reason: "not-marv-root",
      steps: [],
      durationMs: Date.now() - startedAt,
    };
  }

  if (gitRoot) {
    const beforeSha = await getGitHeadSha(runCommand, gitRoot, timeoutMs);
    const beforeVersion = await readPackageVersion(gitRoot);
    const deployStatePath = resolveDeployStatePath(gitRoot);
    const deployState = await readDeployState({ root: gitRoot });
    const lastKnownGoodSha = deployState?.lastKnownGood?.sha ?? null;
    const channel: UpdateChannel = opts.channel ?? "dev";
    const approvalPolicy = normalizeUpdateApprovalConfig(opts.approval);
    const trackedBranch = approvalPolicy?.branch ?? DEV_BRANCH;
    const branch = channel === "dev" ? await readBranchName(runCommand, gitRoot, timeoutMs) : null;
    const needsCheckoutTrackedBranch = channel === "dev" && branch !== trackedBranch;
    gitTotalSteps = channel === "dev" ? (needsCheckoutTrackedBranch ? 16 : 15) : 14;
    let deployTargetSha: string | null = null;
    const buildGitErrorResult = (
      reason: string,
      extra?: { rolledBackToSha?: string | null },
    ): UpdateRunResult => ({
      status: "error",
      mode: "git",
      root: gitRoot,
      reason,
      before: { sha: beforeSha, version: beforeVersion },
      deploy: {
        statePath: deployStatePath,
        lastKnownGoodSha,
        targetSha: deployTargetSha,
        rolledBackToSha: extra?.rolledBackToSha ?? null,
      },
      steps,
      durationMs: Date.now() - startedAt,
    });
    const runGitCheckoutOrFail = async (name: string, argv: string[]) => {
      const checkoutStep = await runStep(step(name, argv, gitRoot));
      steps.push(checkoutStep);
      if (checkoutStep.exitCode !== 0) {
        return buildGitErrorResult("checkout-failed");
      }
      return null;
    };

    const statusCheck = await runStep(
      step(
        "clean check",
        ["git", "-C", gitRoot, "status", "--porcelain", "--", ":!dist/control-ui/"],
        gitRoot,
      ),
    );
    steps.push(statusCheck);
    const hasUncommittedChanges =
      statusCheck.stdoutTail && statusCheck.stdoutTail.trim().length > 0;
    if (hasUncommittedChanges) {
      return {
        status: "skipped",
        mode: "git",
        root: gitRoot,
        reason: "dirty",
        before: { sha: beforeSha, version: beforeVersion },
        steps,
        durationMs: Date.now() - startedAt,
      };
    }

    if (channel === "dev") {
      if (needsCheckoutTrackedBranch) {
        const failure = await runGitCheckoutOrFail(`git checkout ${trackedBranch}`, [
          "git",
          "-C",
          gitRoot,
          "checkout",
          trackedBranch,
        ]);
        if (failure) {
          return failure;
        }
      }

      const fetchStep = await runStep(
        step("git fetch", ["git", "-C", gitRoot, "fetch", "--all", "--prune", "--tags"], gitRoot),
      );
      steps.push(fetchStep);
      if (fetchStep.exitCode !== 0) {
        return {
          status: "error",
          mode: "git",
          root: gitRoot,
          reason: "fetch-failed",
          before: { sha: beforeSha, version: beforeVersion },
          steps,
          durationMs: Date.now() - startedAt,
        };
      }

      let preflightBaseSha: string | null = null;
      let preflightCandidates: string[] = [];

      if (approvalPolicy) {
        const approval = await resolveDeployApproval({
          runCommand,
          root: gitRoot,
          timeoutMs,
          config: approvalPolicy,
        });
        if (!approval.approvedSha) {
          return {
            status: "skipped",
            mode: "git",
            root: gitRoot,
            reason: approval.reason ?? "approval-required",
            before: { sha: beforeSha, version: beforeVersion },
            steps,
            durationMs: Date.now() - startedAt,
          };
        }
        if (beforeSha && approval.approvedSha === beforeSha) {
          return {
            status: "skipped",
            mode: "git",
            root: gitRoot,
            reason: "already-approved",
            before: { sha: beforeSha, version: beforeVersion },
            steps,
            durationMs: Date.now() - startedAt,
          };
        }
        preflightBaseSha = approval.approvedSha;
        preflightCandidates = [approval.approvedSha];
      } else {
        const upstreamStep = await runStep(
          step(
            "upstream check",
            [
              "git",
              "-C",
              gitRoot,
              "rev-parse",
              "--abbrev-ref",
              "--symbolic-full-name",
              "@{upstream}",
            ],
            gitRoot,
          ),
        );
        steps.push(upstreamStep);
        if (upstreamStep.exitCode !== 0) {
          return {
            status: "skipped",
            mode: "git",
            root: gitRoot,
            reason: "no-upstream",
            before: { sha: beforeSha, version: beforeVersion },
            steps,
            durationMs: Date.now() - startedAt,
          };
        }

        const upstreamShaStep = await runStep(
          step(
            "git rev-parse @{upstream}",
            ["git", "-C", gitRoot, "rev-parse", "@{upstream}"],
            gitRoot,
          ),
        );
        steps.push(upstreamShaStep);
        const upstreamSha = upstreamShaStep.stdoutTail?.trim();
        if (!upstreamShaStep.stdoutTail || !upstreamSha) {
          return {
            status: "error",
            mode: "git",
            root: gitRoot,
            reason: "no-upstream-sha",
            before: { sha: beforeSha, version: beforeVersion },
            steps,
            durationMs: Date.now() - startedAt,
          };
        }

        const revListStep = await runStep(
          step(
            "git rev-list",
            ["git", "-C", gitRoot, "rev-list", `--max-count=${PREFLIGHT_MAX_COMMITS}`, upstreamSha],
            gitRoot,
          ),
        );
        steps.push(revListStep);
        if (revListStep.exitCode !== 0) {
          return {
            status: "error",
            mode: "git",
            root: gitRoot,
            reason: "preflight-revlist-failed",
            before: { sha: beforeSha, version: beforeVersion },
            steps,
            durationMs: Date.now() - startedAt,
          };
        }

        preflightBaseSha = upstreamSha;
        preflightCandidates = (revListStep.stdoutTail ?? "")
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean);
      }

      if (!preflightBaseSha || preflightCandidates.length === 0) {
        return {
          status: "error",
          mode: "git",
          root: gitRoot,
          reason: "preflight-no-candidates",
          before: { sha: beforeSha, version: beforeVersion },
          steps,
          durationMs: Date.now() - startedAt,
        };
      }

      const manager = await detectPackageManager(gitRoot);
      const preflight = await runPreflight({
        runCommand,
        gitRoot,
        manager,
        timeoutMs,
        progress,
        baseSha: preflightBaseSha,
        candidates: preflightCandidates,
        stepIndexStart: stepIndex,
        totalSteps: gitTotalSteps,
        trustCi: opts.trustCi,
      });
      steps.push(...preflight.steps);
      stepIndex += preflight.stepCount;

      if (!preflight.selectedSha) {
        return {
          status: "error",
          mode: "git",
          root: gitRoot,
          reason: "preflight-no-good-commit",
          before: { sha: beforeSha, version: beforeVersion },
          steps,
          durationMs: Date.now() - startedAt,
        };
      }

      deployTargetSha = preflight.selectedSha;
      await beginDeployAttempt({
        root: gitRoot,
        trigger: "manual",
        fromSha: beforeSha,
        targetSha: preflight.selectedSha,
      });

      const rebaseStep = await runStep(
        step("git rebase", ["git", "-C", gitRoot, "rebase", preflight.selectedSha], gitRoot),
      );
      steps.push(rebaseStep);
      if (rebaseStep.exitCode !== 0) {
        const abortResult = await runCommand(["git", "-C", gitRoot, "rebase", "--abort"], {
          cwd: gitRoot,
          timeoutMs,
        });
        steps.push({
          name: "git rebase --abort",
          command: "git rebase --abort",
          cwd: gitRoot,
          durationMs: 0,
          exitCode: abortResult.code,
          stdoutTail: trimLogTail(abortResult.stdout, MAX_LOG_CHARS),
          stderrTail: trimLogTail(abortResult.stderr, MAX_LOG_CHARS),
        });
        await finishDeployAttempt({
          root: gitRoot,
          status: "error",
          reason: "rebase-failed",
        }).catch(() => undefined);
        return {
          status: "error",
          mode: "git",
          root: gitRoot,
          reason: "rebase-failed",
          before: { sha: beforeSha, version: beforeVersion },
          steps,
          durationMs: Date.now() - startedAt,
        };
      }
    } else {
      const fetchStep = await runStep(
        step("git fetch", ["git", "-C", gitRoot, "fetch", "--all", "--prune", "--tags"], gitRoot),
      );
      steps.push(fetchStep);
      if (fetchStep.exitCode !== 0) {
        return {
          status: "error",
          mode: "git",
          root: gitRoot,
          reason: "fetch-failed",
          before: { sha: beforeSha, version: beforeVersion },
          steps,
          durationMs: Date.now() - startedAt,
        };
      }

      const tag = await resolveChannelTag(runCommand, gitRoot, timeoutMs, channel);
      if (!tag) {
        return {
          status: "error",
          mode: "git",
          root: gitRoot,
          reason: "no-release-tag",
          before: { sha: beforeSha, version: beforeVersion },
          steps,
          durationMs: Date.now() - startedAt,
        };
      }

      const failure = await runGitCheckoutOrFail(`git checkout ${tag}`, [
        "git",
        "-C",
        gitRoot,
        "checkout",
        "--detach",
        tag,
      ]);
      if (failure) {
        return failure;
      }
      deployTargetSha = await getGitHeadSha(runCommand, gitRoot, timeoutMs);
      await beginDeployAttempt({
        root: gitRoot,
        trigger: "manual",
        fromSha: beforeSha,
        targetSha: deployTargetSha,
      });
    }

    const manager = await detectPackageManager(gitRoot);
    const rollbackTargetSha = lastKnownGoodSha ?? beforeSha;
    const rollbackIfNeeded = async (reason: string): Promise<UpdateRunResult> => {
      const currentSha = await getGitHeadSha(runCommand, gitRoot, timeoutMs);
      if (!rollbackTargetSha || !currentSha || currentSha === rollbackTargetSha) {
        await finishDeployAttempt({
          root: gitRoot,
          status: "error",
          reason,
        }).catch(() => undefined);
        return buildGitErrorResult(reason);
      }
      const rollback = await rollbackGitCheckout({
        runCommand,
        root: gitRoot,
        timeoutMs,
        manager,
        progress,
        steps,
        targetSha: rollbackTargetSha,
        totalSteps: gitTotalSteps,
        startIndex: stepIndex,
      });
      stepIndex += 5;
      await recordDeployRollback({
        root: gitRoot,
        status: rollback.ok ? "ok" : "error",
        fromSha: currentSha,
        toSha: rollback.rolledBackToSha,
        reason,
      }).catch(() => undefined);
      await finishDeployAttempt({
        root: gitRoot,
        status: "error",
        reason,
        rolledBackToSha: rollback.rolledBackToSha,
      }).catch(() => undefined);
      return buildGitErrorResult(reason, {
        rolledBackToSha: rollback.rolledBackToSha,
      });
    };

    const depsStep = await runStep(step("deps install", managerInstallArgs(manager), gitRoot));
    steps.push(depsStep);
    if (depsStep.exitCode !== 0) {
      return await rollbackIfNeeded("deps-install-failed");
    }

    const buildStep = await runStep(step("build", managerScriptArgs(manager, "build"), gitRoot));
    steps.push(buildStep);
    if (buildStep.exitCode !== 0) {
      return await rollbackIfNeeded("build-failed");
    }

    const uiBuildStep = await runStep(
      step("ui:build", managerScriptArgs(manager, "ui:build"), gitRoot),
    );
    steps.push(uiBuildStep);
    if (uiBuildStep.exitCode !== 0) {
      return await rollbackIfNeeded("ui-build-failed");
    }

    const doctorEntryCandidates = [path.join(gitRoot, "marv.mjs"), path.join(gitRoot, "marv.mjs")];
    const doctorEntry =
      (
        await Promise.all(
          doctorEntryCandidates.map(async (candidate) => {
            const exists = await fs
              .stat(candidate)
              .then(() => true)
              .catch(() => false);
            return exists ? candidate : null;
          }),
        )
      ).find(Boolean) ?? null;

    if (!doctorEntry) {
      steps.push({
        name: "marv doctor entry",
        command: `verify ${doctorEntryCandidates.join(" | ")}`,
        cwd: gitRoot,
        durationMs: 0,
        exitCode: 1,
        stderrTail: `missing ${doctorEntryCandidates.join(" and ")}`,
      });
      return await rollbackIfNeeded("doctor-entry-missing");
    }

    // Use --fix so that doctor auto-strips unknown config keys introduced by
    // schema changes between versions, preventing a startup validation crash.
    const doctorArgv = [process.execPath, doctorEntry, "doctor", "--non-interactive", "--fix"];
    const doctorStep = await runStep(
      step("marv doctor", doctorArgv, gitRoot, {
        MARV_UPDATE_IN_PROGRESS: "1",
      }),
    );
    steps.push(doctorStep);

    const uiIndexHealth = await resolveControlUiDistIndexHealth({ root: gitRoot });
    if (!uiIndexHealth.exists) {
      const repairArgv = managerScriptArgs(manager, "ui:build");
      const started = Date.now();
      const repairResult = await runCommand(repairArgv, { cwd: gitRoot, timeoutMs });
      const repairStep: UpdateStepResult = {
        name: "ui:build (post-doctor repair)",
        command: repairArgv.join(" "),
        cwd: gitRoot,
        durationMs: Date.now() - started,
        exitCode: repairResult.code,
        stdoutTail: trimLogTail(repairResult.stdout, MAX_LOG_CHARS),
        stderrTail: trimLogTail(repairResult.stderr, MAX_LOG_CHARS),
      };
      steps.push(repairStep);

      if (repairResult.code !== 0) {
        return await rollbackIfNeeded(repairStep.name);
      }

      const repairedUiIndexHealth = await resolveControlUiDistIndexHealth({ root: gitRoot });
      if (!repairedUiIndexHealth.exists) {
        const uiIndexPath =
          repairedUiIndexHealth.indexPath ?? resolveControlUiDistIndexPathForRoot(gitRoot);
        steps.push({
          name: "ui assets verify",
          command: `verify ${uiIndexPath}`,
          cwd: gitRoot,
          durationMs: 0,
          exitCode: 1,
          stderrTail: `missing ${uiIndexPath}`,
        });
        return await rollbackIfNeeded("ui-assets-missing");
      }
    }

    const failedStep = steps.find((s) => s.exitCode !== 0);
    const afterShaStep = await runStep(
      step("git rev-parse HEAD (after)", ["git", "-C", gitRoot, "rev-parse", "HEAD"], gitRoot),
    );
    steps.push(afterShaStep);
    const afterVersion = await readPackageVersion(gitRoot);

    if (failedStep) {
      return await rollbackIfNeeded(failedStep.name);
    }

    await finishDeployAttempt({
      root: gitRoot,
      status: "ok",
    }).catch(() => undefined);

    return {
      status: "ok",
      mode: "git",
      root: gitRoot,
      reason: undefined,
      before: { sha: beforeSha, version: beforeVersion },
      after: {
        sha: afterShaStep.stdoutTail?.trim() ?? null,
        version: afterVersion,
      },
      deploy: {
        statePath: deployStatePath,
        lastKnownGoodSha,
        targetSha: deployTargetSha,
        rolledBackToSha: null,
      },
      steps,
      durationMs: Date.now() - startedAt,
    };
  }

  if (!pkgRoot) {
    return {
      status: "error",
      mode: "unknown",
      reason: `no root (${START_DIRS.join(",")})`,
      steps: [],
      durationMs: Date.now() - startedAt,
    };
  }

  const beforeVersion = await readPackageVersion(pkgRoot);
  const globalManager = await detectGlobalInstallManagerForRoot(runCommand, pkgRoot, timeoutMs);
  if (globalManager) {
    const packageName = (await readPackageName(pkgRoot)) ?? DEFAULT_PACKAGE_NAME;
    await cleanupGlobalRenameDirs({
      globalRoot: path.dirname(pkgRoot),
      packageName,
    });
    const channel = opts.channel ?? DEFAULT_PACKAGE_CHANNEL;
    const tag = normalizeTag(opts.tag ?? channelToNpmTag(channel));
    const spec = `${packageName}@${tag}`;
    const updateStep = await runStep({
      runCommand,
      name: "global update",
      argv: globalInstallArgs(globalManager, spec),
      cwd: pkgRoot,
      timeoutMs,
      progress,
      stepIndex: 0,
      totalSteps: 1,
    });
    const steps = [updateStep];
    const afterVersion = await readPackageVersion(pkgRoot);
    return {
      status: updateStep.exitCode === 0 ? "ok" : "error",
      mode: globalManager,
      root: pkgRoot,
      reason: updateStep.exitCode === 0 ? undefined : updateStep.name,
      before: { version: beforeVersion },
      after: { version: afterVersion },
      steps,
      durationMs: Date.now() - startedAt,
    };
  }

  return {
    status: "skipped",
    mode: "unknown",
    root: pkgRoot,
    reason: "not-git-install",
    before: { version: beforeVersion },
    steps: [],
    durationMs: Date.now() - startedAt,
  };
}

// ── Rollback ──────────────────────────────────────────────────────────

export async function runGatewayRollback(opts: UpdateRunnerOptions = {}): Promise<UpdateRunResult> {
  const startedAt = Date.now();
  const runCommand: CommandRunner =
    opts.runCommand ??
    (async (argv, options) => {
      const res = await runCommandWithTimeout(argv, options);
      return { stdout: res.stdout, stderr: res.stderr, code: res.code };
    });
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const progress = opts.progress;
  const steps: UpdateStepResult[] = [];
  const dirCandidates = buildStartDirs(opts);
  const gitRoot = await resolveGitRoot(runCommand, dirCandidates, timeoutMs);
  if (!gitRoot || !(await isCorePackageRoot(gitRoot))) {
    return {
      status: "error",
      mode: "unknown",
      reason: "not-git-install",
      steps,
      durationMs: Date.now() - startedAt,
    };
  }

  const beforeSha = await getGitHeadSha(runCommand, gitRoot, timeoutMs);
  const beforeVersion = await readPackageVersion(gitRoot);
  const deployStatePath = resolveDeployStatePath(gitRoot);
  const deployState = await readDeployState({ root: gitRoot });
  const rollbackTargetSha = deployState?.lastKnownGood?.sha ?? null;
  if (!rollbackTargetSha) {
    return {
      status: "skipped",
      mode: "git",
      root: gitRoot,
      reason: "no-last-known-good",
      before: { sha: beforeSha, version: beforeVersion },
      deploy: {
        statePath: deployStatePath,
        lastKnownGoodSha: null,
        targetSha: null,
        rolledBackToSha: null,
      },
      steps,
      durationMs: Date.now() - startedAt,
    };
  }

  const manager = await detectPackageManager(gitRoot);
  const rollback = await rollbackGitCheckout({
    runCommand,
    root: gitRoot,
    timeoutMs,
    manager,
    progress,
    steps,
    targetSha: rollbackTargetSha,
    totalSteps: 5,
    startIndex: 0,
  });
  const afterSha = await getGitHeadSha(runCommand, gitRoot, timeoutMs);
  const afterVersion = await readPackageVersion(gitRoot);

  await recordDeployRollback({
    root: gitRoot,
    status: rollback.ok ? "ok" : "error",
    fromSha: beforeSha,
    toSha: rollback.rolledBackToSha,
    reason: rollback.ok ? undefined : "rollback-failed",
  }).catch(() => undefined);
  await finishDeployAttempt({
    root: gitRoot,
    status: rollback.ok ? "ok" : "error",
    reason: rollback.ok ? "manual-rollback" : "rollback-failed",
    rolledBackToSha: rollback.rolledBackToSha,
  }).catch(() => undefined);

  return {
    status: rollback.ok ? "ok" : "error",
    mode: "git",
    root: gitRoot,
    reason: rollback.ok ? undefined : "rollback-failed",
    before: { sha: beforeSha, version: beforeVersion },
    after: { sha: afterSha, version: afterVersion },
    deploy: {
      statePath: deployStatePath,
      lastKnownGoodSha: rollbackTargetSha,
      targetSha: rollbackTargetSha,
      rolledBackToSha: rollback.rolledBackToSha,
    },
    steps,
    durationMs: Date.now() - startedAt,
  };
}

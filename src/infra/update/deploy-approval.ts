type CommandRunner = (
  argv: string[],
  options: { cwd?: string; timeoutMs?: number },
) => Promise<{ stdout: string; stderr: string; code: number | null }>;

export type UpdateApprovalConfig = {
  required?: boolean;
  mode?: "signed-tag";
  tagPattern?: string;
  branch?: string;
  requireReachableFromBranch?: boolean;
};

export type ResolvedUpdateApprovalConfig = {
  required: true;
  mode: "signed-tag";
  tagPattern: string;
  branch: string;
  requireReachableFromBranch: boolean;
};

export type DeployApprovalStatus = {
  required: true;
  mode: "signed-tag";
  tagPattern: string;
  branch: string;
  branchRef: string | null;
  branchSha: string | null;
  approvedTag: string | null;
  approvedSha: string | null;
  pendingCommits: number | null;
  reason: string | null;
};

const DEFAULT_DEPLOY_TAG_PATTERN = "deploy/*";
const DEFAULT_DEPLOY_BRANCH = "main";

function normalizeNonEmpty(value?: string | null): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
}

async function runGit(
  runCommand: CommandRunner,
  root: string,
  timeoutMs: number,
  argv: string[],
): Promise<{ stdout: string; stderr: string; code: number | null } | null> {
  return await runCommand(["git", "-C", root, ...argv], { cwd: root, timeoutMs }).catch(() => null);
}

async function resolveExistingRef(
  runCommand: CommandRunner,
  root: string,
  timeoutMs: number,
  ref: string,
): Promise<string | null> {
  const res = await runGit(runCommand, root, timeoutMs, [
    "rev-parse",
    "--verify",
    `${ref}^{commit}`,
  ]);
  if (!res || res.code !== 0) {
    return null;
  }
  return ref;
}

async function resolveTrackedBranchRef(params: {
  runCommand: CommandRunner;
  root: string;
  timeoutMs: number;
  branch: string;
}): Promise<string | null> {
  const { runCommand, root, timeoutMs, branch } = params;
  const upstreamRes = await runGit(runCommand, root, timeoutMs, [
    "rev-parse",
    "--abbrev-ref",
    "--symbolic-full-name",
    "@{upstream}",
  ]);
  const upstream = normalizeNonEmpty(upstreamRes?.code === 0 ? upstreamRes.stdout : null);
  if (upstream) {
    const upstreamBranch = upstream.split("/").at(-1) ?? upstream;
    if (upstream === branch || upstreamBranch === branch) {
      const verifiedUpstream = await resolveExistingRef(runCommand, root, timeoutMs, upstream);
      if (verifiedUpstream) {
        return verifiedUpstream;
      }
    }
  }

  const originRef = `origin/${branch}`;
  const verifiedOrigin = await resolveExistingRef(runCommand, root, timeoutMs, originRef);
  if (verifiedOrigin) {
    return verifiedOrigin;
  }

  return await resolveExistingRef(runCommand, root, timeoutMs, branch);
}

export function normalizeUpdateApprovalConfig(
  value?: UpdateApprovalConfig | null,
): ResolvedUpdateApprovalConfig | null {
  if (value?.required !== true) {
    return null;
  }
  return {
    required: true,
    mode: "signed-tag",
    tagPattern: normalizeNonEmpty(value.tagPattern) ?? DEFAULT_DEPLOY_TAG_PATTERN,
    branch: normalizeNonEmpty(value.branch) ?? DEFAULT_DEPLOY_BRANCH,
    requireReachableFromBranch: value.requireReachableFromBranch !== false,
  };
}

export async function resolveDeployApproval(params: {
  runCommand: CommandRunner;
  root: string;
  timeoutMs: number;
  config: ResolvedUpdateApprovalConfig;
}): Promise<DeployApprovalStatus> {
  const { runCommand, root, timeoutMs, config } = params;
  const branchRef = await resolveTrackedBranchRef({
    runCommand,
    root,
    timeoutMs,
    branch: config.branch,
  });
  const branchShaRes =
    branchRef == null ? null : await runGit(runCommand, root, timeoutMs, ["rev-parse", branchRef]);
  const branchSha =
    branchShaRes && branchShaRes.code === 0 ? normalizeNonEmpty(branchShaRes.stdout) : null;

  const base: DeployApprovalStatus = {
    required: true,
    mode: "signed-tag",
    tagPattern: config.tagPattern,
    branch: config.branch,
    branchRef,
    branchSha,
    approvedTag: null,
    approvedSha: null,
    pendingCommits: null,
    reason: null,
  };

  const tagsRes = await runGit(runCommand, root, timeoutMs, [
    "tag",
    "--list",
    config.tagPattern,
    "--sort=-creatordate",
  ]);
  const tags =
    tagsRes && tagsRes.code === 0
      ? tagsRes.stdout
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean)
      : [];
  if (tags.length === 0) {
    return { ...base, reason: "no-matching-tags" };
  }

  if (config.requireReachableFromBranch && !branchRef) {
    return { ...base, reason: "branch-ref-missing" };
  }

  let lastReason = "no-verified-tags";
  for (const tag of tags) {
    const verifyRes = await runGit(runCommand, root, timeoutMs, ["verify-tag", tag]);
    if (!verifyRes || verifyRes.code !== 0) {
      lastReason = "verify-tag-failed";
      continue;
    }

    const shaRes = await runGit(runCommand, root, timeoutMs, ["rev-parse", `${tag}^{commit}`]);
    const approvedSha = shaRes && shaRes.code === 0 ? normalizeNonEmpty(shaRes.stdout) : null;
    if (!approvedSha) {
      lastReason = "tag-target-missing";
      continue;
    }

    if (config.requireReachableFromBranch && branchRef) {
      const reachableRes = await runGit(runCommand, root, timeoutMs, [
        "merge-base",
        "--is-ancestor",
        approvedSha,
        branchRef,
      ]);
      if (!reachableRes || reachableRes.code !== 0) {
        lastReason = "tag-not-on-branch";
        continue;
      }
    }

    let pendingCommits: number | null = null;
    if (branchRef) {
      const pendingRes = await runGit(runCommand, root, timeoutMs, [
        "rev-list",
        "--count",
        `${approvedSha}..${branchRef}`,
      ]);
      const rawPending =
        pendingRes && pendingRes.code === 0 ? Number.parseInt(pendingRes.stdout.trim(), 10) : null;
      pendingCommits = Number.isFinite(rawPending) ? rawPending : null;
    }

    return {
      ...base,
      approvedTag: tag,
      approvedSha,
      pendingCommits,
      reason: null,
    };
  }

  return {
    ...base,
    reason: lastReason,
  };
}

import { describe, expect, it } from "vitest";
import { normalizeUpdateApprovalConfig, resolveDeployApproval } from "./deploy-approval.js";

type CommandResult = { stdout?: string; stderr?: string; code?: number | null };

function createRunner(responses: Record<string, CommandResult>) {
  return async (argv: string[]) => {
    const key = argv.join(" ");
    const result = responses[key] ?? {};
    return {
      stdout: result.stdout ?? "",
      stderr: result.stderr ?? "",
      code: result.code ?? 0,
    };
  };
}

describe("deploy approval", () => {
  it("normalizes required signed-tag approval config", () => {
    expect(normalizeUpdateApprovalConfig({ required: true })).toEqual({
      required: true,
      mode: "signed-tag",
      tagPattern: "deploy/*",
      branch: "main",
      requireReachableFromBranch: true,
    });
  });

  it("resolves the newest verified deploy tag on the tracked branch", async () => {
    const root = "/tmp/marv";
    const runCommand = createRunner({
      [`git -C ${root} rev-parse --abbrev-ref --symbolic-full-name @{upstream}`]: {
        stdout: "origin/main\n",
      },
      [`git -C ${root} rev-parse --verify origin/main^{commit}`]: {
        stdout: "branchsha\n",
      },
      [`git -C ${root} rev-parse origin/main`]: { stdout: "branchsha\n" },
      [`git -C ${root} tag --list deploy/* --sort=-creatordate`]: {
        stdout: "deploy/2026-03-09-2\ndeploy/2026-03-09-1\n",
      },
      [`git -C ${root} verify-tag deploy/2026-03-09-2`]: { code: 1, stderr: "bad signature" },
      [`git -C ${root} verify-tag deploy/2026-03-09-1`]: { stdout: "" },
      [`git -C ${root} rev-parse deploy/2026-03-09-1^{commit}`]: { stdout: "approved123\n" },
      [`git -C ${root} merge-base --is-ancestor approved123 origin/main`]: { code: 0 },
      [`git -C ${root} rev-list --count approved123..origin/main`]: { stdout: "3\n" },
    });

    const approval = await resolveDeployApproval({
      runCommand: async (argv, _opts) => runCommand(argv),
      root,
      timeoutMs: 5_000,
      config: normalizeUpdateApprovalConfig({ required: true })!,
    });

    expect(approval).toMatchObject({
      approvedTag: "deploy/2026-03-09-1",
      approvedSha: "approved123",
      branchRef: "origin/main",
      branchSha: "branchsha",
      pendingCommits: 3,
      reason: null,
    });
  });
});

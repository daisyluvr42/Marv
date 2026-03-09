import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  beginDeployAttempt,
  finishDeployAttempt,
  markDeployHealthy,
  readDeployState,
  resolveDeployStatePath,
} from "./deploy-state.js";

describe("deploy-state", () => {
  const cleanupDirs: string[] = [];

  afterEach(async () => {
    vi.unstubAllEnvs();
    await Promise.all(
      cleanupDirs.splice(0).map((dir) => fs.rm(dir, { recursive: true, force: true })),
    );
  });

  it("stores deploy state under the configured state dir", async () => {
    const stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-deploy-state-"));
    cleanupDirs.push(stateDir);
    vi.stubEnv("MARV_STATE_DIR", stateDir);

    const statePath = resolveDeployStatePath("/tmp/marv");
    expect(statePath.startsWith(path.join(stateDir, "update-deploy"))).toBe(true);
  });

  it("promotes the current revision to lastKnownGood on healthy startup", async () => {
    const stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-deploy-state-"));
    const repoDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-deploy-repo-"));
    cleanupDirs.push(stateDir, repoDir);
    vi.stubEnv("MARV_STATE_DIR", stateDir);
    await fs.writeFile(
      path.join(repoDir, "package.json"),
      JSON.stringify({ name: "agentmarv", version: "1.2.3" }),
      "utf-8",
    );

    const runCommand = vi.fn(async (argv: string[]) => {
      const key = argv.join(" ");
      if (key === `git -C ${repoDir} rev-parse HEAD`) {
        return { stdout: "abc123\n", stderr: "", code: 0 };
      }
      if (key === `git -C ${repoDir} rev-parse --abbrev-ref HEAD`) {
        return { stdout: "main\n", stderr: "", code: 0 };
      }
      if (key === `git -C ${repoDir} rev-parse --abbrev-ref --symbolic-full-name @{upstream}`) {
        return { stdout: "origin/main\n", stderr: "", code: 0 };
      }
      return { stdout: "", stderr: "", code: 1 };
    });

    await markDeployHealthy({ root: repoDir, runCommand });
    const state = await readDeployState({ root: repoDir });

    expect(state?.current?.sha).toBe("abc123");
    expect(state?.lastKnownGood?.sha).toBe("abc123");
    expect(state?.current?.version).toBe("1.2.3");
  });

  it("records deploy attempts and completion state", async () => {
    const stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-deploy-state-"));
    cleanupDirs.push(stateDir);
    vi.stubEnv("MARV_STATE_DIR", stateDir);

    await beginDeployAttempt({
      root: "/tmp/marv",
      trigger: "manual",
      fromSha: "old123",
      targetSha: "new456",
    });
    await finishDeployAttempt({
      root: "/tmp/marv",
      status: "error",
      reason: "build-failed",
      rolledBackToSha: "old123",
    });

    const state = await readDeployState({ root: "/tmp/marv" });
    expect(state?.lastAttempt?.trigger).toBe("manual");
    expect(state?.lastAttempt?.fromSha).toBe("old123");
    expect(state?.lastAttempt?.targetSha).toBe("new456");
    expect(state?.lastAttempt?.status).toBe("error");
    expect(state?.lastAttempt?.rolledBackToSha).toBe("old123");
  });
});

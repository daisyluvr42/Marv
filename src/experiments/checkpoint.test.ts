import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  FileCopyCheckpointStrategy,
  GitCheckpointStrategy,
  JsonSnapshotStrategy,
  NoRollbackStrategy,
  resolveCheckpointStrategy,
} from "./checkpoint.js";

// ── GitCheckpointStrategy ───────────────────────────────────────────

describe("GitCheckpointStrategy", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = path.join(os.tmpdir(), `marv-git-checkpoint-test-${Date.now()}`);
    await fs.mkdir(tmpDir, { recursive: true });
    // Init a git repo
    const { exec } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const execAsync = promisify(exec);
    await execAsync("git init", { cwd: tmpDir });
    await execAsync('git config user.email "test@test.com"', { cwd: tmpDir });
    await execAsync('git config user.name "test"', { cwd: tmpDir });
    // Create initial commit
    await fs.writeFile(path.join(tmpDir, "file.txt"), "initial");
    await execAsync("git add -A && git commit -m 'initial'", { cwd: tmpDir });
  });

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
  });

  it("saves and restores via git commits", async () => {
    const strategy = new GitCheckpointStrategy(tmpDir);

    // Make a change and save checkpoint
    await fs.writeFile(path.join(tmpDir, "file.txt"), "modified");
    const ref = await strategy.save("test-checkpoint");
    expect(ref.strategy).toBe("git");
    expect(ref.ref).toMatch(/^[a-f0-9]+$/);

    // Make another change
    await fs.writeFile(path.join(tmpDir, "file.txt"), "further-modified");

    // Restore
    await strategy.restore(ref);
    const content = await fs.readFile(path.join(tmpDir, "file.txt"), "utf-8");
    expect(content).toBe("modified");
  });

  it("handles clean working tree (no changes to commit)", async () => {
    const strategy = new GitCheckpointStrategy(tmpDir);
    const ref = await strategy.save("clean");
    expect(ref.strategy).toBe("git");
  });

  it("describe returns useful info", () => {
    const strategy = new GitCheckpointStrategy("/some/path");
    expect(strategy.describe()).toContain("/some/path");
  });
});

// ── FileCopyCheckpointStrategy ──────────────────────────────────────

describe("FileCopyCheckpointStrategy", () => {
  let tmpDir: string;
  let testFile: string;

  beforeEach(async () => {
    tmpDir = path.join(os.tmpdir(), `marv-filecopy-test-${Date.now()}`);
    await fs.mkdir(tmpDir, { recursive: true });
    testFile = path.join(tmpDir, "config.json");
    await fs.writeFile(testFile, JSON.stringify({ key: "original" }));
  });

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
  });

  it("saves and restores file contents", async () => {
    const strategy = new FileCopyCheckpointStrategy([testFile]);

    const ref = await strategy.save("test");
    expect(ref.strategy).toBe("file-copy");

    // Modify the file
    await fs.writeFile(testFile, JSON.stringify({ key: "modified" }));

    // Restore
    await strategy.restore(ref);
    const content = JSON.parse(await fs.readFile(testFile, "utf-8"));
    expect(content.key).toBe("original");
  });

  it("handles files that do not exist at save time", async () => {
    const newFile = path.join(tmpDir, "new-file.json");
    const strategy = new FileCopyCheckpointStrategy([newFile]);

    const ref = await strategy.save("absent");

    // Create the file after checkpoint
    await fs.writeFile(newFile, "should be removed");

    // Restore — file should be removed since it was absent at save time
    await strategy.restore(ref);
    await expect(fs.access(newFile)).rejects.toThrow();
  });

  it("describe shows file count", () => {
    const strategy = new FileCopyCheckpointStrategy(["/a", "/b", "/c"]);
    expect(strategy.describe()).toContain("3 files");
  });
});

// ── JsonSnapshotStrategy ────────────────────────────────────────────

describe("JsonSnapshotStrategy", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = path.join(os.tmpdir(), `marv-json-snapshot-test-${Date.now()}`);
    await fs.mkdir(tmpDir, { recursive: true });
  });

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
  });

  it("saves and reads JSON snapshots", async () => {
    const strategy = new JsonSnapshotStrategy(tmpDir);
    const ref = await strategy.save("state-1");

    const data = { config: { model: "gpt-4", temperature: 0.7 } };
    await strategy.writeSnapshot(ref, data);

    const restored = await strategy.readSnapshot(ref);
    expect(restored).toEqual(data);
  });

  it("restore is a no-op (application-specific)", async () => {
    const strategy = new JsonSnapshotStrategy(tmpDir);
    const ref = await strategy.save("test");
    // Should not throw
    await strategy.restore(ref);
  });
});

// ── NoRollbackStrategy ──────────────────────────────────────────────

describe("NoRollbackStrategy", () => {
  it("save returns a ref with strategy=none", async () => {
    const strategy = new NoRollbackStrategy();
    const ref = await strategy.save("test");
    expect(ref.strategy).toBe("none");
  });

  it("restore is a no-op", async () => {
    const strategy = new NoRollbackStrategy();
    const ref = await strategy.save("test");
    // Should not throw
    await strategy.restore(ref);
  });

  it("describe is informative", () => {
    const strategy = new NoRollbackStrategy();
    expect(strategy.describe()).toContain("no-rollback");
  });
});

// ── resolveCheckpointStrategy ───────────────────────────────────────

describe("resolveCheckpointStrategy", () => {
  it("resolves git strategy", () => {
    const strategy = resolveCheckpointStrategy({ strategy: "git", cwd: "/tmp" });
    expect(strategy).toBeInstanceOf(GitCheckpointStrategy);
  });

  it("resolves file-copy strategy", () => {
    const strategy = resolveCheckpointStrategy({ strategy: "file-copy", paths: ["/a", "/b"] });
    expect(strategy).toBeInstanceOf(FileCopyCheckpointStrategy);
  });

  it("resolves json-snapshot strategy", () => {
    const strategy = resolveCheckpointStrategy({ strategy: "json-snapshot" });
    expect(strategy).toBeInstanceOf(JsonSnapshotStrategy);
  });

  it("resolves none strategy", () => {
    const strategy = resolveCheckpointStrategy({ strategy: "none" });
    expect(strategy).toBeInstanceOf(NoRollbackStrategy);
  });
});

import { exec } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { CheckpointConfig, CheckpointRef, CheckpointStrategy } from "./types.js";

// ── Git Checkpoint ──────────────────────────────────────────────────

export class GitCheckpointStrategy implements CheckpointStrategy {
  constructor(private readonly cwd: string = process.cwd()) {}

  async save(label: string): Promise<CheckpointRef> {
    // Get current HEAD before any changes
    const headBefore = (await git(this.cwd, "rev-parse", "HEAD")).trim();

    // Create an auto-commit checkpoint (only if there are staged/unstaged changes)
    const status = await git(this.cwd, "status", "--porcelain");
    if (status.trim()) {
      await git(this.cwd, "add", "-A");
      await git(this.cwd, "commit", "-m", `experiment checkpoint: ${label}`, "--allow-empty");
    }

    const commitHash = (await git(this.cwd, "rev-parse", "HEAD")).trim();
    return {
      strategy: "git",
      ref: commitHash,
      metadata: { headBefore, label },
    };
  }

  async restore(ref: CheckpointRef): Promise<void> {
    await git(this.cwd, "reset", "--hard", ref.ref);
  }

  describe(): string {
    return `git checkpoint (cwd: ${this.cwd})`;
  }
}

// ── File Copy Checkpoint ────────────────────────────────────────────

export class FileCopyCheckpointStrategy implements CheckpointStrategy {
  private readonly backupRoot: string;

  constructor(private readonly paths: string[]) {
    this.backupRoot = path.join(os.tmpdir(), `marv-experiment-backup-${Date.now()}`);
  }

  async save(label: string): Promise<CheckpointRef> {
    const backupDir = path.join(this.backupRoot, label);
    await fs.mkdir(backupDir, { recursive: true });

    // Back up each file
    for (const filePath of this.paths) {
      const resolved = filePath.startsWith("~")
        ? path.join(os.homedir(), filePath.slice(1))
        : path.resolve(filePath);

      try {
        const dest = path.join(backupDir, path.basename(resolved));
        await fs.copyFile(resolved, dest);
      } catch {
        // File doesn't exist yet — record that it was absent
        const marker = path.join(backupDir, `${path.basename(resolved)}.absent`);
        await fs.writeFile(marker, "");
      }
    }

    return {
      strategy: "file-copy",
      ref: backupDir,
      metadata: { label, paths: this.paths },
    };
  }

  async restore(ref: CheckpointRef): Promise<void> {
    const backupDir = ref.ref;

    for (const filePath of this.paths) {
      const resolved = filePath.startsWith("~")
        ? path.join(os.homedir(), filePath.slice(1))
        : path.resolve(filePath);

      const backedUp = path.join(backupDir, path.basename(resolved));
      const absentMarker = `${backedUp}.absent`;

      try {
        await fs.access(absentMarker);
        // File was absent before — remove it
        await fs.rm(resolved, { force: true });
      } catch {
        // Restore from backup
        try {
          await fs.mkdir(path.dirname(resolved), { recursive: true });
          await fs.copyFile(backedUp, resolved);
        } catch {
          // Backup file missing — nothing to restore
        }
      }
    }
  }

  describe(): string {
    return `file-copy checkpoint (${this.paths.length} files)`;
  }
}

// ── JSON Snapshot Checkpoint ────────────────────────────────────────

export class JsonSnapshotStrategy implements CheckpointStrategy {
  private readonly snapshotDir: string;

  constructor(snapshotDir?: string) {
    this.snapshotDir =
      snapshotDir ?? path.join(os.tmpdir(), `marv-experiment-snapshots-${Date.now()}`);
  }

  async save(label: string): Promise<CheckpointRef> {
    await fs.mkdir(this.snapshotDir, { recursive: true });
    const snapshotPath = path.join(this.snapshotDir, `${label}.json`);
    // The actual state capture is delegated to the caller via metadata.
    // This strategy just provides the storage location.
    return {
      strategy: "json-snapshot",
      ref: snapshotPath,
      metadata: { label },
    };
  }

  /** Write snapshot data to the ref path. Call this after save() with the actual state. */
  async writeSnapshot(ref: CheckpointRef, data: unknown): Promise<void> {
    await fs.writeFile(ref.ref, JSON.stringify(data, null, 2));
  }

  /** Read snapshot data from the ref path. */
  async readSnapshot(ref: CheckpointRef): Promise<unknown> {
    const raw = await fs.readFile(ref.ref, "utf-8");
    return JSON.parse(raw);
  }

  async restore(_ref: CheckpointRef): Promise<void> {
    // JSON snapshot restore is application-specific.
    // The caller reads the snapshot via readSnapshot() and applies it.
    // This is a no-op at the strategy level.
  }

  describe(): string {
    return `json-snapshot checkpoint (dir: ${this.snapshotDir})`;
  }
}

// ── No Rollback Strategy ────────────────────────────────────────────

export class NoRollbackStrategy implements CheckpointStrategy {
  async save(label: string): Promise<CheckpointRef> {
    return {
      strategy: "none",
      ref: `no-rollback-${label}`,
      metadata: {
        label,
        note: "No checkpoint created. Changes cannot be automatically rolled back.",
      },
    };
  }

  async restore(_ref: CheckpointRef): Promise<void> {
    // No-op: just log that rollback was requested but not possible.
    // The experiment loop handles this by recording it in the iteration log.
  }

  describe(): string {
    return "no-rollback (changes logged only)";
  }
}

// ── Factory ─────────────────────────────────────────────────────────

/** Create a CheckpointStrategy from a config object. */
export function resolveCheckpointStrategy(config: CheckpointConfig): CheckpointStrategy {
  switch (config.strategy) {
    case "git":
      return new GitCheckpointStrategy(config.cwd);
    case "file-copy":
      return new FileCopyCheckpointStrategy(config.paths);
    case "json-snapshot":
      return new JsonSnapshotStrategy(config.snapshotDir);
    case "none":
      return new NoRollbackStrategy();
  }
}

// ── Git Helper ──────────────────────────────────────────────────────

function git(cwd: string, ...args: string[]): Promise<string> {
  const escaped = args.map((a) => `'${a.replace(/'/g, "'\\''")}'`).join(" ");
  return new Promise((resolve, reject) => {
    exec(`git ${escaped}`, { cwd, timeout: 30_000 }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(`git ${args[0]}: ${stderr || error.message}`));
        return;
      }
      resolve(stdout);
    });
  });
}

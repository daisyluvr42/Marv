import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { clearConfigCache } from "../core/config/config.js";
import { createNonExitingRuntime } from "../runtime.js";
import { migrateExportCommand } from "./migrate-export.js";
import { migrateImportCommand } from "./migrate-import.js";

const ENV_KEYS = [
  "HOME",
  "USERPROFILE",
  "MARV_HOME",
  "MARV_STATE_DIR",
  "MARV_CONFIG_PATH",
  "MARV_OAUTH_DIR",
] as const;

const originalEnv = new Map<string, string | undefined>(
  ENV_KEYS.map((key) => [key, process.env[key]]),
);

afterEach(() => {
  for (const [key, value] of originalEnv.entries()) {
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
  clearConfigCache();
  vi.restoreAllMocks();
});

describe("migrate export/import", () => {
  it("exports and imports the selected scopes in non-interactive mode", async () => {
    const sourceHome = await fs.mkdtemp(path.join(os.tmpdir(), "marv-migrate-source-"));
    const targetHome = await fs.mkdtemp(path.join(os.tmpdir(), "marv-migrate-target-"));
    const archivePath = path.join(os.tmpdir(), `marv-migrate-${Date.now().toString(36)}.tar.gz`);

    await seedTestState(sourceHome);

    process.env.HOME = sourceHome;
    delete process.env.MARV_HOME;
    delete process.env.MARV_STATE_DIR;
    delete process.env.MARV_CONFIG_PATH;
    delete process.env.MARV_OAUTH_DIR;
    clearConfigCache();

    const exportRuntime = createNonExitingRuntime();
    vi.spyOn(exportRuntime, "log").mockImplementation(() => {});
    vi.spyOn(exportRuntime, "error").mockImplementation(() => {});

    await migrateExportCommand(exportRuntime, {
      scopes: ["memory", "config", "sessions", "credentials", "workspace", "tasks", "ledger"],
      format: "plain",
      output: archivePath,
      nonInteractive: true,
    });

    process.env.HOME = targetHome;
    clearConfigCache();

    const importRuntime = createNonExitingRuntime();
    vi.spyOn(importRuntime, "log").mockImplementation(() => {});
    vi.spyOn(importRuntime, "error").mockImplementation(() => {});

    await migrateImportCommand(importRuntime, {
      archivePath,
      force: true,
      nonInteractive: true,
    });

    const targetState = path.join(targetHome, ".marv");
    await expect(
      fs.readFile(path.join(targetState, "memory", "soul", "main.sqlite"), "utf-8"),
    ).resolves.toBe("memory-db");
    await expect(fs.readFile(path.join(targetState, "marv.json"), "utf-8")).resolves.toContain(
      '"workspace":"~/.marv/workspace"',
    );
    await expect(
      fs.readFile(path.join(targetState, "agents", "main", "sessions", "chat.jsonl"), "utf-8"),
    ).resolves.toContain("hello");
    await expect(
      fs.readFile(path.join(targetState, "agents", "main", "sessions.json"), "utf-8"),
    ).resolves.toContain("chat");
    await expect(
      fs.readFile(path.join(targetState, "credentials", "token.json"), "utf-8"),
    ).resolves.toContain("secret");
    await expect(
      fs.readFile(path.join(targetState, "workspace", "AGENTS.md"), "utf-8"),
    ).resolves.toContain("workspace agent rules");
    await expect(
      fs.readFile(path.join(targetState, "tasks", "task.json"), "utf-8"),
    ).resolves.toContain("ship");
    await expect(
      fs.readFile(path.join(targetState, "ledger", "events.sqlite"), "utf-8"),
    ).resolves.toBe("ledger-db");
  });

  it("supports dry-run imports without overwriting existing files", async () => {
    const sourceHome = await fs.mkdtemp(path.join(os.tmpdir(), "marv-migrate-source-"));
    const targetHome = await fs.mkdtemp(path.join(os.tmpdir(), "marv-migrate-target-"));
    const archivePath = path.join(os.tmpdir(), `marv-migrate-${Date.now().toString(36)}.tar.gz`);

    await seedTestState(sourceHome);

    process.env.HOME = sourceHome;
    clearConfigCache();

    const exportRuntime = createNonExitingRuntime();
    vi.spyOn(exportRuntime, "log").mockImplementation(() => {});
    vi.spyOn(exportRuntime, "error").mockImplementation(() => {});

    await migrateExportCommand(exportRuntime, {
      scopes: ["config"],
      format: "plain",
      output: archivePath,
      nonInteractive: true,
    });

    const targetState = path.join(targetHome, ".marv");
    await fs.mkdir(targetState, { recursive: true });
    await fs.writeFile(
      path.join(targetState, "marv.json"),
      '{"agents":{"defaults":{"workspace":"~/.marv/existing-workspace"}}}\n',
    );

    process.env.HOME = targetHome;
    clearConfigCache();

    const importRuntime = createNonExitingRuntime();
    vi.spyOn(importRuntime, "log").mockImplementation(() => {});
    vi.spyOn(importRuntime, "error").mockImplementation(() => {});

    await migrateImportCommand(importRuntime, {
      archivePath,
      dryRun: true,
      nonInteractive: true,
    });

    await expect(fs.readFile(path.join(targetState, "marv.json"), "utf-8")).resolves.toBe(
      '{"agents":{"defaults":{"workspace":"~/.marv/existing-workspace"}}}\n',
    );
  });
});

async function seedTestState(homeDir: string) {
  const stateDir = path.join(homeDir, ".marv");
  await fs.mkdir(path.join(stateDir, "memory", "soul"), { recursive: true });
  await fs.mkdir(path.join(stateDir, "agents", "main", "sessions"), { recursive: true });
  await fs.mkdir(path.join(stateDir, "credentials"), { recursive: true });
  await fs.mkdir(path.join(stateDir, "workspace"), { recursive: true });
  await fs.mkdir(path.join(stateDir, "tasks"), { recursive: true });
  await fs.mkdir(path.join(stateDir, "ledger"), { recursive: true });

  await fs.writeFile(
    path.join(stateDir, "marv.json"),
    '{"agents":{"defaults":{"workspace":"~/.marv/workspace"}}}\n',
  );
  await fs.writeFile(path.join(stateDir, "memory", "soul", "main.sqlite"), "memory-db");
  await fs.writeFile(
    path.join(stateDir, "agents", "main", "sessions", "chat.jsonl"),
    '{"role":"user","content":"hello"}\n',
  );
  await fs.writeFile(path.join(stateDir, "agents", "main", "sessions.json"), '{"last":"chat"}\n');
  await fs.writeFile(path.join(stateDir, "credentials", "token.json"), '{"token":"secret"}\n');
  await fs.writeFile(path.join(stateDir, "workspace", "AGENTS.md"), "workspace agent rules\n");
  await fs.writeFile(path.join(stateDir, "tasks", "task.json"), '{"task":"ship"}\n');
  await fs.writeFile(path.join(stateDir, "ledger", "events.sqlite"), "ledger-db");
}

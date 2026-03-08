import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { exportSoulMemoryMirror, importSoulMemoryMirror } from "./soul-memory-mirror.js";
import { listSoulMemoryItems, writeSoulMemory } from "./soul-memory-store.js";

const ORIGINAL_STATE_DIR = process.env.MARV_STATE_DIR;

let stateDir = "";
let workspaceDir = "";

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-mirror-state-"));
  workspaceDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-mirror-workspace-"));
  process.env.MARV_STATE_DIR = stateDir;
});

afterEach(async () => {
  if (ORIGINAL_STATE_DIR === undefined) {
    delete process.env.MARV_STATE_DIR;
  } else {
    process.env.MARV_STATE_DIR = ORIGINAL_STATE_DIR;
  }
  if (stateDir) {
    await fs.rm(stateDir, { recursive: true, force: true });
  }
  if (workspaceDir) {
    await fs.rm(workspaceDir, { recursive: true, force: true });
  }
});

describe("soul-memory-mirror", () => {
  it("exports selected structured memory into MEMORY.md", async () => {
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "Prefer concise replies.",
      source: "core_preference",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "policy",
      content: "Keep changelog entries user-facing.",
      source: "manual_log",
    });

    const result = await exportSoulMemoryMirror({
      agentId: "main",
      workspaceDir,
    });
    expect(result.updated).toBe(true);
    expect(result.requiresConfirmation).toBe(false);

    const text = await fs.readFile(path.join(workspaceDir, "MEMORY.md"), "utf-8");
    expect(text).toContain("## P0");
    expect(text).toContain("Prefer concise replies.");
    expect(text).toContain("## Stable P1");
  });

  it("pauses export and shows conflicts when MEMORY.md has unsynced manual edits", async () => {
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "Prefer concise replies.",
      source: "core_preference",
    });

    await exportSoulMemoryMirror({
      agentId: "main",
      workspaceDir,
    });

    const mirrorPath = path.join(workspaceDir, "MEMORY.md");
    const original = await fs.readFile(mirrorPath, "utf-8");
    const edited = original.replace(
      "Prefer concise replies.",
      "Prefer concise replies and bullet lists.",
    );
    await fs.writeFile(mirrorPath, edited, "utf-8");

    const blocked = await exportSoulMemoryMirror({
      agentId: "main",
      workspaceDir,
    });
    expect(blocked.updated).toBe(false);
    expect(blocked.requiresConfirmation).toBe(true);
    expect(blocked.conflicts).toContainEqual({
      tier: "P0",
      kind: "preference",
      content: "Prefer concise replies and bullet lists.",
    });
  });

  it("imports explicit mirror edits back into structured memory and refreshes the mirror hash", async () => {
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "Prefer concise replies.",
      source: "core_preference",
    });

    await exportSoulMemoryMirror({
      agentId: "main",
      workspaceDir,
    });

    const mirrorPath = path.join(workspaceDir, "MEMORY.md");
    const original = await fs.readFile(mirrorPath, "utf-8");
    const edited = original.replace(
      "Prefer concise replies.",
      "Prefer concise replies and bullet lists.",
    );
    await fs.writeFile(mirrorPath, edited, "utf-8");

    const imported = await importSoulMemoryMirror({
      agentId: "main",
      workspaceDir,
    });
    expect(imported.imported).toBeGreaterThan(0);

    const memories = listSoulMemoryItems({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      limit: 20,
    });
    expect(memories.some((item) => item.content.includes("bullet lists"))).toBe(true);

    const refreshed = await fs.readFile(mirrorPath, "utf-8");
    expect(refreshed).toContain("Prefer concise replies and bullet lists.");
  });
});

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { detectSoulMemoryConflicts, listSoulMemoryConflicts } from "./soul-memory-conflict.js";
import { writeSoulMemory } from "./soul-memory-store.js";

let stateDir = "";
let previousStateDir: string | undefined;

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-soul-conflict-"));
  previousStateDir = process.env.MARV_STATE_DIR;
  process.env.MARV_STATE_DIR = stateDir;
});

afterEach(async () => {
  if (previousStateDir === undefined) {
    delete process.env.MARV_STATE_DIR;
  } else {
    process.env.MARV_STATE_DIR = previousStateDir;
  }
  if (stateDir) {
    await fs.rm(stateDir, { recursive: true, force: true });
  }
});

describe("detectSoulMemoryConflicts", () => {
  it("inserts unresolved conflicts for contradictory memories and avoids duplicates", () => {
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "policy",
      content: "Always use const declarations in JavaScript projects.",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "policy",
      content: "Never use const declarations in JavaScript projects.",
      source: "manual_log",
    });

    const first = detectSoulMemoryConflicts({
      agentId: "main",
      minConfidence: 0.5,
      overlapThreshold: 0.2,
    });
    expect(first.inserted).toBe(1);
    expect(first.conflicts).toHaveLength(1);
    expect(first.conflicts[0]?.conflictReason).toContain("opposite");

    const second = detectSoulMemoryConflicts({
      agentId: "main",
      minConfidence: 0.5,
      overlapThreshold: 0.2,
    });
    expect(second.inserted).toBe(0);

    const unresolved = listSoulMemoryConflicts({
      agentId: "main",
      unresolvedOnly: true,
      limit: 10,
    });
    expect(unresolved).toHaveLength(1);
    expect(unresolved[0]?.resolvedAt).toBeUndefined();
  });
});

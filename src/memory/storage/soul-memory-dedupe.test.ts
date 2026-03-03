import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { dedupeSoulMemories } from "./soul-memory-dedupe.js";
import { listSoulMemoryItems, writeSoulMemory } from "./soul-memory-store.js";

let stateDir = "";
let previousStateDir: string | undefined;

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-soul-dedupe-"));
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

describe("dedupeSoulMemories", () => {
  it("merges highly similar memories and keeps reinforcement on the survivor", () => {
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "Always use const declarations in JavaScript.",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "Always use const declarations in JavaScript",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "Prefer short commit messages.",
      source: "manual_log",
    });

    const dedupe = dedupeSoulMemories({
      agentId: "main",
      similarityThreshold: 0.9,
    });
    expect(dedupe.mergedPairs).toBe(1);
    expect(dedupe.removedIds).toHaveLength(1);

    const items = listSoulMemoryItems({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      limit: 50,
    });
    expect(items).toHaveLength(2);
    const constPref = items.find((item) =>
      item.content.toLowerCase().includes("const declarations"),
    );
    expect(constPref?.reinforcementCount).toBeGreaterThanOrEqual(2);
  });
});

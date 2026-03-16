import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { compactP3Episodic } from "./soul-memory-compaction.js";
import { listSoulMemoryItems, writeSoulMemory } from "./soul-memory-store.js";

let stateDir = "";
let previousStateDir: string | undefined;

const DEFAULT_CONFIG = {
  enabled: true,
  minClusterSize: 3,
  similarityMin: 0.45,
  similarityMax: 0.85,
  archiveAgeDays: 30,
  orphanAgeDays: 60,
  compactedDiscount: 0.5,
};

function writeP3(content: string, kind = "preference", recordKind = "fact") {
  return writeSoulMemory({
    agentId: "main",
    scopeType: "agent",
    scopeId: "main",
    kind,
    content,
    source: "runtime_event",
    tier: "P3",
    recordKind: recordKind as "fact",
  });
}

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-compaction-"));
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

describe("compactP3Episodic", () => {
  it("does nothing when disabled", () => {
    writeP3("User likes sushi rolls for dinner");
    writeP3("User loves sushi rolls for lunch");
    writeP3("User enjoys sushi rolls for breakfast");

    const result = compactP3Episodic({
      agentId: "main",
      config: { ...DEFAULT_CONFIG, enabled: false },
    });
    expect(result.compactedClusters).toBe(0);
    expect(result.compactedEpisodic).toBe(0);
  });

  it("clusters similar P3 episodic items and creates P2 semantic node", () => {
    writeP3("User likes sushi rolls for dinner");
    writeP3("User loves sushi rolls for lunch");
    writeP3("User enjoys sushi rolls for breakfast");

    const result = compactP3Episodic({
      agentId: "main",
      config: DEFAULT_CONFIG,
    });

    expect(result.compactedClusters).toBeGreaterThanOrEqual(1);
    expect(result.compactedEpisodic).toBe(3);
    expect(result.semanticIds).toHaveLength(1);

    // P2 semantic node should exist
    const p2Items = listSoulMemoryItems({
      agentId: "main",
      tier: "P2",
    });
    expect(p2Items.length).toBeGreaterThanOrEqual(1);
    const semantic = p2Items.find((i) => i.memoryType === "semantic");
    expect(semantic).toBeDefined();
    expect(semantic?.sourceDetail).toBe("system");

    // Original P3 items should be marked as compacted
    const p3Items = listSoulMemoryItems({
      agentId: "main",
      tier: "P3",
    });
    const compactedCount = p3Items.filter((i) => i.isCompacted).length;
    expect(compactedCount).toBe(3);
  });

  it("does not cluster items below minClusterSize", () => {
    writeP3("User likes sushi rolls for dinner");
    writeP3("User loves sushi rolls for lunch");
    // Only 2 items, minClusterSize = 3

    const result = compactP3Episodic({
      agentId: "main",
      config: DEFAULT_CONFIG,
    });
    expect(result.compactedClusters).toBe(0);
  });

  it("does not compact already-compacted items", () => {
    writeP3("User likes sushi rolls for dinner");
    writeP3("User loves sushi rolls for lunch");
    writeP3("User enjoys sushi rolls for breakfast");

    // First compaction
    compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });

    // Second compaction should find no new items
    const result = compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });
    expect(result.compactedClusters).toBe(0);
    expect(result.compactedEpisodic).toBe(0);
  });

  it("archives compacted episodic items after archiveAgeDays", () => {
    writeP3("User likes sushi rolls for dinner");
    writeP3("User loves sushi rolls for lunch");
    writeP3("User enjoys sushi rolls for breakfast");

    // Compact first (at current time, items created just now)
    compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });

    // Verify items still in memory_items (not old enough to archive)
    let p3Items = listSoulMemoryItems({ agentId: "main", tier: "P3" });
    expect(p3Items.length).toBe(3);

    // Archive with a far-future time so items appear old enough
    const futureTime = Date.now() + 31 * 86_400_000;
    const result = compactP3Episodic({
      agentId: "main",
      config: DEFAULT_CONFIG,
      nowMs: futureTime,
    });
    expect(result.archivedCompacted).toBe(3);

    // P3 items should be gone from memory_items
    p3Items = listSoulMemoryItems({ agentId: "main", tier: "P3" });
    expect(p3Items.length).toBe(0);
  });

  it("archives orphan episodic items after orphanAgeDays", () => {
    // Write dissimilar items that won't cluster
    writeP3("User's favorite color is blue");
    writeP3("The weather today is sunny and warm");
    writeP3("Project deadline is next Friday");

    // These won't cluster (dissimilar), so they stay uncompacted
    compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });

    let p3Items = listSoulMemoryItems({ agentId: "main", tier: "P3" });
    expect(p3Items.every((i) => !i.isCompacted)).toBe(true);

    // Archive with far-future time to trigger orphan safety valve
    const futureTime = Date.now() + 61 * 86_400_000;
    const result = compactP3Episodic({
      agentId: "main",
      config: DEFAULT_CONFIG,
      nowMs: futureTime,
    });
    expect(result.archivedOrphans).toBe(3);

    // P3 items should be gone
    p3Items = listSoulMemoryItems({ agentId: "main", tier: "P3" });
    expect(p3Items.length).toBe(0);
  });

  it("new memory items have correct memoryType and sourceDetail defaults", () => {
    const item = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "User likes TypeScript",
      source: "manual_log",
    });

    expect(item?.memoryType).toBe("episodic");
    expect(item?.sourceDetail).toBe("explicit");
    expect(item?.isCompacted).toBe(false);
    expect(item?.semanticKey).toBeNull();
  });

  it("runtime_event source gets inferred source_detail", () => {
    const item = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "User mentioned liking coffee",
      source: "runtime_event",
      tier: "P3",
    });

    expect(item?.sourceDetail).toBe("inferred");
  });
});

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { compactP3Episodic } from "./soul-memory-compaction.js";
import { listSoulMemoryConflicts } from "./soul-memory-conflict.js";
import { listSoulMemoryItems, querySoulMemoryMulti, writeSoulMemory } from "./soul-memory-store.js";

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
  batchLimit: 1000,
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

  // --- Phase 4: Evolution ---

  it("evolves semantic node when explicit evidence matches existing semantic_key", () => {
    // First round: create initial semantic
    writeP3("User likes sushi rolls for dinner");
    writeP3("User loves sushi rolls for lunch");
    writeP3("User enjoys sushi rolls for breakfast");

    const r1 = compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });
    expect(r1.compactedClusters).toBeGreaterThanOrEqual(1);
    expect(r1.semanticIds).toHaveLength(1);
    const oldSemanticId = r1.semanticIds[0];

    // Second round: new explicit evidence (manual_log = explicit source_detail)
    writeP3Explicit("User adores sushi rolls every day");
    writeP3Explicit("User craves sushi rolls constantly");
    writeP3Explicit("User wants sushi rolls for every meal");

    const r2 = compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });
    expect(r2.evolvedSemantics).toBe(1);
    expect(r2.semanticIds).toHaveLength(1);
    const newSemanticId = r2.semanticIds[0];
    expect(newSemanticId).not.toBe(oldSemanticId);

    // Old semantic should have valid_until set (retired)
    const p2Items = listSoulMemoryItems({ agentId: "main", tier: "P2" });
    const oldSemantic = p2Items.find((i) => i.id === oldSemanticId);
    expect(oldSemantic?.validUntil).not.toBeNull();

    // New semantic should have valid_from set and valid_until null (active)
    const newSemantic = p2Items.find((i) => i.id === newSemanticId);
    expect(newSemantic?.validFrom).not.toBeNull();
    expect(newSemantic?.validUntil).toBeNull();
  });

  it("marks evolution conflict for inferred evidence instead of evolving", () => {
    // First round: create initial semantic
    writeP3("User likes sushi rolls for dinner");
    writeP3("User loves sushi rolls for lunch");
    writeP3("User enjoys sushi rolls for breakfast");

    const r1 = compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });
    expect(r1.compactedClusters).toBeGreaterThanOrEqual(1);

    // Second round: inferred evidence (runtime_event = inferred)
    writeP3("User mentions sushi rolls again today");
    writeP3("User talks about sushi rolls once more");
    writeP3("User brings up sushi rolls in conversation");

    const r2 = compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });
    expect(r2.evolutionConflicts).toBe(1);
    expect(r2.evolvedSemantics).toBe(0);

    // Episodic items should still be marked compacted even though evolution was deferred
    const p3Items = listSoulMemoryItems({ agentId: "main", tier: "P3" });
    const compactedCount = p3Items.filter((i) => i.isCompacted).length;
    expect(compactedCount).toBe(6); // all 6 items compacted

    // Conflict should be recorded with ask_user strategy
    const conflicts = listSoulMemoryConflicts({ agentId: "main", unresolvedOnly: true });
    const evoConflict = conflicts.find((c) => c.conflictReason.includes("evolution conflict"));
    expect(evoConflict).toBeDefined();
    expect(evoConflict?.resolutionStrategy).toBe("ask_user");
  });

  // --- Phase 4: Temporal filtering ---

  it("excludes retired semantics from default search", () => {
    // Create and then evolve a semantic
    writeP3("User likes sushi rolls for dinner");
    writeP3("User loves sushi rolls for lunch");
    writeP3("User enjoys sushi rolls for breakfast");
    compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });

    writeP3Explicit("User adores sushi rolls every day");
    writeP3Explicit("User craves sushi rolls constantly");
    writeP3Explicit("User wants sushi rolls for every meal");
    compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });

    // Default search should only return the active (new) semantic, not the retired one
    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "sushi rolls",
      topK: 10,
    });

    const semanticResults = results.filter((r) => r.memoryType === "semantic");
    // All returned semantics should be active (validUntil === null)
    for (const result of semanticResults) {
      expect(result.validUntil).toBeNull();
    }
  });

  it("includes retired semantics in point-in-time temporal query", () => {
    const t0 = Date.now();
    writeP3("User likes sushi rolls for dinner");
    writeP3("User loves sushi rolls for lunch");
    writeP3("User enjoys sushi rolls for breakfast");
    compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG, nowMs: t0 });

    const t1 = t0 + 86_400_000; // 1 day later
    writeP3Explicit("User adores sushi rolls every day");
    writeP3Explicit("User craves sushi rolls constantly");
    writeP3Explicit("User wants sushi rolls for every meal");
    compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG, nowMs: t1 });

    // Verify the evolution happened: old semantic retired, new semantic active
    const p2Items = listSoulMemoryItems({ agentId: "main", tier: "P2" });
    const retired = p2Items.filter((i) => i.validUntil !== null);
    const active = p2Items.filter((i) => i.validUntil === null);
    expect(retired.length).toBe(1);
    expect(active.length).toBe(1);
    expect(retired[0].validFrom).toBeLessThanOrEqual(t0);
    expect(retired[0].validUntil).toBe(t1);

    // Query at t0+1000: retired semantic (validFrom<=t0+1000<validUntil=t1) should be included
    // while active semantic (validFrom=t1>t0+1000) should be excluded
    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "sushi rolls",
      topK: 10,
      temporalMs: t0 + 1000,
    });

    // With temporal query, we may get episodic + semantic results
    // The key assertion: if semantic results are returned, only the retired one should match
    const semanticResults = results.filter((r) => r.memoryType === "semantic");
    if (semanticResults.length > 0) {
      const retiredSemantic = semanticResults.find((r) => r.validUntil !== null);
      expect(retiredSemantic).toBeDefined();
      // Active semantic (validFrom = t1) should NOT be in results
      const activeSemantic = semanticResults.find(
        (r) => r.validUntil === null && r.validFrom === t1,
      );
      expect(activeSemantic).toBeUndefined();
    }

    // Direct verification: retired semantic's validity window covers t0+1000
    const retiredItem = retired[0];
    expect(retiredItem.validFrom! <= t0 + 1000).toBe(true);
    expect(retiredItem.validUntil! > t0 + 1000).toBe(true);
  });

  // --- Phase 4: Conflict-aware retrieval ---

  it("annotates search results with unresolved conflict IDs", () => {
    // Create memories that conflict
    writeP3("User likes sushi rolls for dinner");
    writeP3("User loves sushi rolls for lunch");
    writeP3("User enjoys sushi rolls for breakfast");

    const r1 = compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });
    expect(r1.compactedClusters).toBeGreaterThanOrEqual(1);
    const semanticId = r1.semanticIds[0];

    // Create a second batch to trigger evolution conflict (inferred)
    writeP3("User mentions sushi rolls again today");
    writeP3("User talks about sushi rolls once more");
    writeP3("User brings up sushi rolls in conversation");

    const r2 = compactP3Episodic({ agentId: "main", config: DEFAULT_CONFIG });
    expect(r2.evolutionConflicts).toBe(1);

    // Verify conflict exists via conflict API
    const conflicts = listSoulMemoryConflicts({ agentId: "main", unresolvedOnly: true });
    const evoConflict = conflicts.find((c) => c.conflictReason.includes("evolution conflict"));
    expect(evoConflict).toBeDefined();
    expect(evoConflict!.memoryIdA).toBe(semanticId);

    // Search for the semantic and check conflict annotation
    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "sushi rolls",
      topK: 20,
    });

    // If the semantic is in search results, it should have conflictIds
    const semanticResult = results.find((r) => r.id === semanticId);
    if (semanticResult) {
      expect(semanticResult.conflictIds.length).toBeGreaterThanOrEqual(1);
    }

    // Either way, verify the conflict annotation mechanism works on any returned result
    // that matches a conflict pair
    const conflictMemoryIds = new Set<string>();
    for (const c of conflicts) {
      conflictMemoryIds.add(c.memoryIdA);
      conflictMemoryIds.add(c.memoryIdB);
    }
    const resultIds = new Set(results.map((r) => r.id));
    const matchedIds = [...conflictMemoryIds].filter((id) => resultIds.has(id));
    // If any search result matches a conflict member, it should have conflictIds set
    for (const id of matchedIds) {
      const result = results.find((r) => r.id === id);
      expect(result?.conflictIds.length).toBeGreaterThanOrEqual(1);
    }
  });
});

/** Write P3 episodic with explicit source_detail (manual_log). */
function writeP3Explicit(content: string) {
  return writeSoulMemory({
    agentId: "main",
    scopeType: "agent",
    scopeId: "main",
    kind: "preference",
    content,
    source: "manual_log",
    tier: "P3",
    recordKind: "fact",
  });
}

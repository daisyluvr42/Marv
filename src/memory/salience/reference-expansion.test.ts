import { DatabaseSync } from "node:sqlite";
import { describe, expect, it } from "vitest";
import {
  applyReferenceExpansion,
  loadReferencesBySourceIds,
  upsertItemReferences,
} from "./reference-expansion.js";

describe("reference-expansion", () => {
  it("extracts and upserts unique refs while ignoring self links", () => {
    const db = new DatabaseSync(":memory:");
    db.exec(
      "CREATE TABLE memory_item_refs (" +
        "source_memory_id TEXT NOT NULL, target_memory_id TEXT NOT NULL, created_at INTEGER NOT NULL, " +
        "PRIMARY KEY (source_memory_id, target_memory_id)" +
        ");",
    );

    upsertItemReferences(db, "mem_a", "links [ref:mem_b] [ref:MEM_B] [ref:mem_a] [ref:mem_c]", 1);
    const refs = loadReferencesBySourceIds(db, ["mem_a"]);
    expect(refs.get("mem_a")).toEqual(["mem_b", "mem_c"]);

    db.close();
  });

  it("applies multi-hop boosts within hop limit", () => {
    const db = new DatabaseSync(":memory:");
    db.exec(
      "CREATE TABLE memory_item_refs (" +
        "source_memory_id TEXT NOT NULL, target_memory_id TEXT NOT NULL, created_at INTEGER NOT NULL, " +
        "PRIMARY KEY (source_memory_id, target_memory_id)" +
        ");",
    );
    db.prepare(
      "INSERT INTO memory_item_refs (source_memory_id, target_memory_id, created_at) VALUES (?, ?, ?)",
    ).run("a", "b", 1);
    db.prepare(
      "INSERT INTO memory_item_refs (source_memory_id, target_memory_id, created_at) VALUES (?, ?, ?)",
    ).run("b", "c", 1);
    db.prepare(
      "INSERT INTO memory_item_refs (source_memory_id, target_memory_id, created_at) VALUES (?, ?, ?)",
    ).run("c", "d", 1);

    const scoredById = new Map([
      ["a", { id: "a", score: 1, referenceBoost: 0 }],
      ["b", { id: "b", score: 0.4, referenceBoost: 0 }],
      ["c", { id: "c", score: 0.2, referenceBoost: 0 }],
      ["d", { id: "d", score: 0.1, referenceBoost: 0 }],
    ]);

    applyReferenceExpansion({
      db,
      scoredById,
      topK: 1,
      soulConfig: {
        referenceExpansionEnabled: true,
        referenceMaxHops: 2,
        referenceEdgeDecay: 0.5,
        referenceBoostWeight: 0.4,
        referenceMaxBoost: 0.6,
        referenceSeedTopKMultiplier: 1,
      },
    });

    expect(scoredById.get("b")?.referenceBoost).toBeCloseTo(0.2, 5);
    expect(scoredById.get("c")?.referenceBoost).toBeCloseTo(0.1, 5);
    expect(scoredById.get("d")?.referenceBoost).toBe(0);
    expect((scoredById.get("b")?.score ?? 0) > 0.4).toBe(true);
    expect((scoredById.get("c")?.score ?? 0) > 0.2).toBe(true);

    db.close();
  });
});

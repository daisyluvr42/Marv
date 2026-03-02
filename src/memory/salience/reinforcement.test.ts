import { DatabaseSync } from "node:sqlite";
import { describe, expect, it } from "vitest";
import {
  computeReinforcementFactor,
  recordScopeHits,
  reinforceRetrievedItems,
} from "./reinforcement.js";

describe("reinforcement", () => {
  it("follows logarithmic reinforcement growth", () => {
    const config = { reinforcementLogWeight: 0.2 };
    const f1 = computeReinforcementFactor(1, config);
    const f2 = computeReinforcementFactor(2, config);
    const f3 = computeReinforcementFactor(3, config);
    const f4 = computeReinforcementFactor(4, config);

    expect(f1).toBe(1);
    expect(f2).toBeGreaterThan(f1);
    expect(f3).toBeGreaterThan(f2);
    expect(f4).toBeGreaterThan(f3);
    expect(f3 - f2).toBeLessThan(f2 - f1);
    expect(computeReinforcementFactor(99, { reinforcementLogWeight: 0 })).toBe(1);
  });

  it("updates memory reinforcement and records deduped scope hits", () => {
    const db = new DatabaseSync(":memory:");
    db.exec(
      "CREATE TABLE memory_items (" +
        "id TEXT PRIMARY KEY, confidence REAL, reinforcement_count INTEGER, " +
        "last_accessed_at INTEGER, last_reinforced_at INTEGER" +
        ");",
    );
    db.exec(
      "CREATE TABLE memory_scope_hits (" +
        "memory_id TEXT NOT NULL, scope_id TEXT NOT NULL, hit_count INTEGER NOT NULL DEFAULT 0, " +
        "first_hit_at INTEGER NOT NULL, last_hit_at INTEGER NOT NULL, " +
        "PRIMARY KEY (memory_id, scope_id)" +
        ");",
    );

    db.prepare(
      "INSERT INTO memory_items (id, confidence, reinforcement_count) VALUES (?, ?, ?)",
    ).run("a", 0.5, 1);
    db.prepare(
      "INSERT INTO memory_items (id, confidence, reinforcement_count) VALUES (?, ?, ?)",
    ).run("b", 0.2, 1);

    reinforceRetrievedItems(
      db,
      [
        { id: "a", wasRecallBoosted: false },
        { id: "a", wasRecallBoosted: false },
        { id: "b", wasRecallBoosted: true },
      ],
      100,
    );

    const a = db
      .prepare(
        "SELECT confidence, reinforcement_count, last_accessed_at FROM memory_items WHERE id = ?",
      )
      .get("a") as
      | {
          confidence?: number;
          reinforcement_count?: number;
          last_accessed_at?: number;
        }
      | undefined;
    const b = db
      .prepare(
        "SELECT confidence, reinforcement_count, last_reinforced_at FROM memory_items WHERE id = ?",
      )
      .get("b") as
      | {
          confidence?: number;
          reinforcement_count?: number;
          last_reinforced_at?: number;
        }
      | undefined;

    expect(a?.confidence).toBeCloseTo(0.55);
    expect(a?.reinforcement_count).toBe(2);
    expect(a?.last_accessed_at).toBe(100);
    expect(b?.confidence).toBe(1);
    expect(b?.reinforcement_count).toBe(2);
    expect(b?.last_reinforced_at).toBe(100);

    recordScopeHits(db, [{ id: "a" }, { id: "a" }, { id: "b" }], "Project:Alpha", 200);
    recordScopeHits(db, [{ id: "a" }], "project:alpha", 250);

    const rows = db
      .prepare(
        "SELECT memory_id, scope_id, hit_count FROM memory_scope_hits ORDER BY memory_id ASC",
      )
      .all() as Array<{
      memory_id?: string;
      scope_id?: string;
      hit_count?: number;
    }>;
    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({ memory_id: "a", scope_id: "project:alpha", hit_count: 2 });
    expect(rows[1]).toMatchObject({ memory_id: "b", scope_id: "project:alpha", hit_count: 1 });

    db.close();
  });
});

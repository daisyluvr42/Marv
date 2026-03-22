import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  applySoulMemoryConfidenceDecay,
  buildSoulArchivePath,
  buildSoulMemoryPath,
  countSoulArchiveEvents,
  getSoulArchiveEvent,
  getSoulMemoryItem,
  ingestSoulMemoryEvent,
  listSoulMemoryItems,
  listSoulMemoryReferences,
  parseSoulArchivePath,
  parseSoulMemoryPath,
  promoteSoulMemories,
  querySoulArchive,
  querySoulMemoryMulti,
  writeSoulMemory,
} from "./soul-memory-store.js";

const ORIGINAL_STATE_DIR = process.env.MARV_STATE_DIR;

let stateDir = "";

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-soul-memory-"));
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
});

describe("soul-memory-store", () => {
  it("writes mapped source/tier and deduplicates identical entries", () => {
    const first = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "请记住我喜欢低糖",
      source: "core_preference",
      confidence: 0.01,
    });
    expect(first).not.toBeNull();
    expect(first?.tier).toBe("P3");
    expect(first?.confidence).toBeCloseTo(0.95);
    expect(first?.reinforcementCount).toBe(1);

    const second = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "请记住我喜欢低糖",
      source: "core_preference",
    });
    expect(second?.id).toBe(first?.id);
    expect(second?.reinforcementCount).toBe(2);
    expect(second?.lastReinforcedAt).not.toBeNull();

    const all = listSoulMemoryItems({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
    });
    expect(all).toHaveLength(1);
  });

  it("increments reinforcement_count on retrieval and applies configurable reinforcement boost", () => {
    const item = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "deploy checklist includes rollback validation",
      source: "manual_log",
    });
    expect(item).not.toBeNull();
    if (!item) {
      throw new Error("item missing");
    }
    expect(item.reinforcementCount).toBe(1);

    const first = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "deploy checklist includes rollback validation",
      topK: 1,
      minScore: 0,
    });
    expect(first).toHaveLength(1);
    expect(first[0]?.reinforcementCount).toBe(1);
    expect(first[0]?.reinforcementFactor).toBeCloseTo(1);

    const second = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "deploy checklist includes rollback validation",
      topK: 1,
      minScore: 0,
    });
    expect(second).toHaveLength(1);
    expect(second[0]?.reinforcementCount).toBe(2);
    expect(second[0]?.reinforcementFactor ?? 0).toBeGreaterThan(1);
    expect((second[0]?.score ?? 0) > (first[0]?.score ?? 0)).toBe(true);

    const noBoost = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "deploy checklist includes rollback validation",
      topK: 1,
      minScore: 0,
      soulConfig: {
        reinforcementLogWeight: 0,
      },
    });
    expect(noBoost).toHaveLength(1);
    expect(noBoost[0]?.reinforcementCount).toBeGreaterThanOrEqual(3);
    expect(noBoost[0]?.reinforcementFactor).toBeCloseTo(1);

    const refreshed = getSoulMemoryItem({ agentId: "main", itemId: item.id });
    expect(refreshed?.reinforcementCount).toBeGreaterThanOrEqual(4);
    expect(refreshed?.lastReinforcedAt).not.toBeNull();
  });

  it("exposes explicit salience score from reinforcement and time decay", () => {
    const createdMs = Date.UTC(2026, 0, 1, 0, 0, 0);
    const item = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "salience tracking memory item",
      source: "manual_log",
      nowMs: createdMs,
    });
    expect(item).not.toBeNull();

    const fresh = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "salience tracking memory item",
      topK: 1,
      minScore: 0,
      nowMs: createdMs + 1 * 24 * 60 * 60 * 1000,
    });
    expect(fresh).toHaveLength(1);
    expect(fresh[0]?.salienceScore ?? 0).toBeGreaterThan(0);
    expect(fresh[0]?.salienceDecay ?? 0).toBeGreaterThan(0);
    expect(fresh[0]?.salienceReinforcement ?? 0).toBeGreaterThanOrEqual(1);

    const stale = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "salience tracking memory item",
      topK: 1,
      minScore: 0,
      nowMs: createdMs + 180 * 24 * 60 * 60 * 1000,
    });
    expect(stale).toHaveLength(1);
    // No decay in the new architecture: salienceDecay is always 1 regardless of age
    expect(stale[0]?.salienceDecay).toBeCloseTo(1);
    expect(fresh[0]?.salienceDecay).toBeCloseTo(1);
  });

  it("tracks [ref:item_id] citation links and returns them in query results", () => {
    const base = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "base decision entry",
      source: "manual_log",
    });
    expect(base).not.toBeNull();
    if (!base) {
      throw new Error("base item missing");
    }

    const linked = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: `follow-up note pointing to [ref:${base.id}]`,
      source: "manual_log",
    });
    expect(linked).not.toBeNull();
    if (!linked) {
      throw new Error("linked item missing");
    }

    const refs = listSoulMemoryReferences({ agentId: "main", itemId: linked.id });
    expect(refs).toEqual([base.id]);

    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "follow-up note pointing",
      topK: 1,
      minScore: 0,
    });
    expect(results).toHaveLength(1);
    expect(results[0]?.references).toContain(base.id);
  });

  it("expands multi-hop references and applies chain-weight boost during search", () => {
    const c = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "core rollback policy anchor",
      source: "manual_log",
    });
    expect(c).not.toBeNull();
    if (!c) {
      throw new Error("c missing");
    }

    const b = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: `intermediate deployment checklist [ref:${c.id}]`,
      source: "manual_log",
    });
    expect(b).not.toBeNull();
    if (!b) {
      throw new Error("b missing");
    }

    const a = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: `urgent hotfix procedure [ref:${b.id}]`,
      source: "manual_log",
    });
    expect(a).not.toBeNull();

    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "urgent hotfix procedure",
      topK: 3,
      minScore: 0,
    });
    expect(results).toHaveLength(3);

    const cResult = results.find((entry) => entry.id === c.id);
    expect(cResult).toBeDefined();
    expect(cResult?.referenceBoost ?? 0).toBeGreaterThan(0);
    expect(cResult?.score ?? 0).toBeGreaterThan(0);
  });

  it("assigns all sources to P3 with no tier-based preference for stale records", () => {
    const oldMs = Date.UTC(2026, 0, 1, 0, 0, 0);
    const nowMs = oldMs + 30 * 24 * 60 * 60 * 1000;
    const p0 = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "alpha preference p0",
      source: "core_preference",
      nowMs: oldMs,
    });
    const p2 = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "alpha preference p2",
      source: "auto_extraction",
      nowMs: oldMs,
    });
    expect(p0).not.toBeNull();
    expect(p2).not.toBeNull();

    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "alpha preference",
      topK: 2,
      minScore: 0,
      ttlDays: 3650,
      nowMs,
    });
    expect(results).toHaveLength(2);
    // All sources map to P3 — no tier-based preference
    expect(results[0]?.tier).toBe("P3");
    expect(results[1]?.tier).toBe("P3");
  });

  it("downgrades core_preference source to manual_log when kind is not soul-level", () => {
    const item = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "ephemeral scratch line that should not enter P0",
      source: "core_preference",
    });
    expect(item).not.toBeNull();
    // All sources map to P3 in the new architecture
    expect(item?.tier).toBe("P3");
    expect(item?.source).toBe("manual_log");
    expect(item?.confidence).toBeCloseTo(0.85);
  });

  it("keeps core_preference source for configured kinds but still assigns P3 tier", () => {
    const item = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "stable canonical project note",
      source: "core_preference",
      soulConfig: {
        p0AllowedKinds: ["note"],
      },
    });
    expect(item).not.toBeNull();
    // All sources map to P3 in the new architecture
    expect(item?.tier).toBe("P3");
    expect(item?.source).toBe("core_preference");
    expect(item?.confidence).toBeCloseTo(0.95);
  });

  it("returns equal time decay for all memories (no decay in new architecture)", () => {
    const baseMs = Date.UTC(2026, 0, 1, 0, 0, 0);
    const nowMs = baseMs + 180 * 24 * 60 * 60 * 1000;
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "incident release checklist old copy",
      source: "manual_log",
      nowMs: baseMs,
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "incident release checklist recent copy",
      source: "manual_log",
      nowMs: baseMs + 150 * 24 * 60 * 60 * 1000,
    });

    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "incident release checklist",
      topK: 2,
      minScore: 0,
      ttlDays: 3650,
      nowMs,
    });
    expect(results).toHaveLength(2);
    // No decay: both items have timeDecay = 1 regardless of age
    expect(results[0]?.timeDecay).toBeCloseTo(1);
    expect(results[1]?.timeDecay).toBeCloseTo(1);
  });

  it("keeps retrieval stable for exact token queries (bm25/fts path)", () => {
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "部署代号 R9ZXQ7 仅用于回滚窗口",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "普通周会纪要",
      source: "manual_log",
    });

    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "R9ZXQ7 + 回滚 (窗口)",
      topK: 2,
      minScore: 0,
    });
    expect(results[0]?.content).toContain("R9ZXQ7");
    expect(results[0]?.rrfScore).toBeGreaterThan(0);
    expect(results[0]?.bm25Score).toBeGreaterThanOrEqual(0);
  });

  it("adds graph and cluster signals for linked memories", () => {
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "alice owns projectphoenix migration",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "projectphoenix runbook key zeta42",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "preference",
      content: "tea preference for mornings",
      source: "manual_log",
    });

    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "alice runbook",
      topK: 3,
      minScore: 0,
    });

    const linked = results.find((entry) => entry.content.includes("projectphoenix runbook"));
    const unrelated = results.find((entry) => entry.content.includes("tea preference"));
    expect(linked).toBeDefined();
    expect((linked?.graphScore ?? 0) > (unrelated?.graphScore ?? 0)).toBe(true);
    expect(linked?.clusterScore).toBeGreaterThan(0);
  });

  it("applies strong cross-scope penalty for unrelated memories", () => {
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "release matrix parity token hub",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "project",
      scopeId: "other",
      kind: "note",
      content: "release matrix parity token hub from unrelated scope",
      source: "manual_log",
    });

    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "release matrix parity token hub",
      topK: 2,
      minScore: 0,
    });
    expect(results).toHaveLength(2);
    expect(results[0]?.scopeId).toBe("main");
    expect(results[0]?.scopePenalty).toBeCloseTo(1);
    expect(results[1]?.scopeId).toBe("other");
    expect(results[1]?.scopePenalty).toBeCloseTo(0.2);
    expect((results[0]?.score ?? 0) > (results[1]?.score ?? 0)).toBe(true);
  });

  it("applies configurable scope penalty coefficients", () => {
    writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "ops runbook sync policy",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "project",
      scopeId: "other",
      kind: "note",
      content: "ops runbook sync policy for project other",
      source: "manual_log",
    });

    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "ops runbook sync policy",
      topK: 2,
      minScore: 0,
      soulConfig: {
        crossScopePenalty: 0.55,
      },
    });
    expect(results).toHaveLength(2);
    const crossScope = results.find((entry) => entry.scopeId === "other");
    expect(crossScope?.scopePenalty).toBeCloseTo(0.55);
  });

  it("no longer boosts dormant memories — all items are P3 with no recall boost", () => {
    const baseMs = Date.UTC(2024, 0, 1, 0, 0, 0);
    const nowMs = baseMs + 720 * 24 * 60 * 60 * 1000;
    const item = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "guardrail",
      content: "never swallow stack traces on failure",
      source: "core_preference",
      nowMs: baseMs,
    });
    expect(item).not.toBeNull();
    if (!item) {
      throw new Error("item missing");
    }

    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "never swallow stack traces on failure",
      topK: 1,
      minScore: 0,
      nowMs,
    });
    expect(results).toHaveLength(1);
    // All sources map to P3, no recall boost in the new architecture
    expect(results[0]?.tier).toBe("P3");
    expect(results[0]?.wasRecallBoosted).toBe(false);
    // clarityScore = confidence directly (no decay); computed before reinforcement bump
    expect(results[0]?.clarityScore).toBeCloseTo(0.95);

    const refreshed = getSoulMemoryItem({ agentId: "main", itemId: item.id });
    // confidence bumped from 0.95 to 1.0 via normal retrieval reinforcement (+0.05)
    expect(refreshed?.confidence).toBeCloseTo(1.0);
  });

  it("promoteSoulMemories is a no-op — no promotion from P2 to P1", () => {
    const createdMs = Date.UTC(2026, 0, 1, 0, 0, 0);
    const nowMs = createdMs + 20 * 24 * 60 * 60 * 1000;
    const item = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "operator escalation playbook redline",
      source: "auto_extraction",
      nowMs: createdMs,
    });
    expect(item).not.toBeNull();
    if (!item) {
      throw new Error("item missing");
    }

    const scopes = ["proj-a", "proj-b", "proj-c"];
    for (let i = 0; i < 8; i += 1) {
      querySoulMemoryMulti({
        agentId: "main",
        scopes: [
          { scopeType: "project", scopeId: scopes[i % scopes.length] ?? "proj-a", weight: 1 },
        ],
        query: "operator escalation playbook redline",
        topK: 1,
        minScore: 0,
        nowMs,
      });
    }

    const summary = promoteSoulMemories({
      agentId: "main",
      nowMs,
    });
    // Promotion is a no-op in the new architecture
    expect(summary.promotedToP1).toBe(0);
    expect(summary.p1PromotionIds).toHaveLength(0);
    expect(summary.skillExtractionCandidates).toHaveLength(0);
    // Item stays at P3
    const unchanged = getSoulMemoryItem({ agentId: "main", itemId: item.id });
    expect(unchanged?.tier).toBe("P3");
  });

  it("promoteSoulMemories emits no approval candidates and performs no P0 promotion", () => {
    const createdMs = Date.UTC(2025, 0, 1, 0, 0, 0);
    const nowMs = createdMs + 315 * 24 * 60 * 60 * 1000;
    const item = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "principle",
      content: "always preserve raw error stack in incident reports",
      source: "manual_log",
      nowMs: createdMs,
    });
    expect(item).not.toBeNull();
    if (!item) {
      throw new Error("item missing");
    }

    for (let day = 30; day <= 300; day += 30) {
      querySoulMemoryMulti({
        agentId: "main",
        scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
        query: "always preserve raw error stack in incident reports",
        topK: 1,
        minScore: 0,
        nowMs: createdMs + day * 24 * 60 * 60 * 1000,
      });
    }

    const pending = promoteSoulMemories({
      agentId: "main",
      nowMs,
    });
    // Promotion is a no-op — no approval candidates emitted
    expect(pending.promotedToP0).toBe(0);
    expect(pending.p0ApprovalCandidates).toHaveLength(0);

    const approved = promoteSoulMemories({
      agentId: "main",
      nowMs,
      approvedP0Ids: [item.id],
    });
    // Even with approved IDs, promotion is a no-op
    expect(approved.promotedToP0).toBe(0);
    expect(approved.p0PromotionIds).toHaveLength(0);
    // Item stays at P3
    const unchanged = getSoulMemoryItem({ agentId: "main", itemId: item.id });
    expect(unchanged?.tier).toBe("P3");
  });

  it("promoteSoulMemories ignores age gating config — always returns empty", () => {
    const createdMs = Date.UTC(2026, 0, 1, 0, 0, 0);
    const nowMs = createdMs + 60 * 24 * 60 * 60 * 1000;
    const item = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "principle",
      content: "always include rollback notes in release docs",
      source: "manual_log",
      nowMs: createdMs,
    });
    expect(item).not.toBeNull();
    if (!item) {
      throw new Error("item missing");
    }

    querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "always include rollback notes in release docs",
      topK: 1,
      minScore: 0,
      nowMs: createdMs + 59 * 24 * 60 * 60 * 1000,
    });

    const defaultSummary = promoteSoulMemories({
      agentId: "main",
      nowMs,
    });
    // No approval candidates — promotion is a no-op
    expect(defaultSummary.p0ApprovalCandidates).toHaveLength(0);

    const configuredSummary = promoteSoulMemories({
      agentId: "main",
      nowMs,
      soulConfig: {
        p1ToP0MinAgeDays: 30,
      },
    });
    // Config is ignored — still no candidates
    expect(configuredSummary.p0ApprovalCandidates).toHaveLength(0);
  });

  it("no pruning — all items are retained regardless of staleness", () => {
    const createdMs = Date.UTC(2024, 0, 1, 0, 0, 0);
    const nowMs = createdMs + 800 * 24 * 60 * 60 * 1000;
    const p0 = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "policy",
      content: "keep privacy sensitive defaults",
      source: "core_preference",
      nowMs: createdMs,
    });
    const p1 = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "obsolete migration scratch note",
      source: "manual_log",
      nowMs: createdMs,
    });
    expect(p0).not.toBeNull();
    expect(p1).not.toBeNull();
    if (!p0 || !p1) {
      throw new Error("item missing");
    }

    const decay = applySoulMemoryConfidenceDecay({
      agentId: "main",
      nowMs,
    });
    // No pruning in the new architecture
    expect(decay.deleted).toBe(0);
    // Both items remain
    expect(getSoulMemoryItem({ agentId: "main", itemId: p1.id })).not.toBeNull();
    expect(getSoulMemoryItem({ agentId: "main", itemId: p0.id })).not.toBeNull();
  });

  it("never prunes items regardless of age or confidence", () => {
    const createdMs = Date.UTC(2026, 0, 1, 0, 0, 0);
    const p2 = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "short lived extraction memory",
      source: "auto_extraction",
      nowMs: createdMs,
    });
    expect(p2).not.toBeNull();
    if (!p2) {
      throw new Error("item missing");
    }

    const at45Days = applySoulMemoryConfidenceDecay({
      agentId: "main",
      nowMs: createdMs + 45 * 24 * 60 * 60 * 1000,
    });
    expect(at45Days.deleted).toBe(0);
    expect(getSoulMemoryItem({ agentId: "main", itemId: p2.id })).not.toBeNull();

    const at60Days = applySoulMemoryConfidenceDecay({
      agentId: "main",
      nowMs: createdMs + 60 * 24 * 60 * 60 * 1000,
    });
    // No pruning in the new architecture — item survives indefinitely
    expect(at60Days.deleted).toBe(0);
    expect(getSoulMemoryItem({ agentId: "main", itemId: p2.id })).not.toBeNull();
  });

  it("pruning config has no effect — items are never pruned", () => {
    const createdMs = Date.UTC(2026, 0, 1, 0, 0, 0);
    const p2 = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "temporary extraction candidate",
      source: "auto_extraction",
      nowMs: createdMs,
    });
    expect(p2).not.toBeNull();
    if (!p2) {
      throw new Error("item missing");
    }

    const baseline = applySoulMemoryConfidenceDecay({
      agentId: "main",
      nowMs: createdMs + 10 * 24 * 60 * 60 * 1000,
    });
    expect(baseline.deleted).toBe(0);

    const aggressive = applySoulMemoryConfidenceDecay({
      agentId: "main",
      nowMs: createdMs + 10 * 24 * 60 * 60 * 1000,
      soulConfig: {
        p2ClarityHalfLifeDays: 2,
        forgetStreakHalfLives: 1,
      },
    });
    // Pruning config is ignored — no items deleted
    expect(aggressive.deleted).toBe(0);
    expect(getSoulMemoryItem({ agentId: "main", itemId: p2.id })).not.toBeNull();
  });

  it("round-trips soul-memory paths", () => {
    const item = writeSoulMemory({
      agentId: "main",
      scopeType: "agent",
      scopeId: "main",
      kind: "note",
      content: "remember this",
      source: "manual_log",
    });
    expect(item).not.toBeNull();
    if (!item) {
      throw new Error("item missing");
    }

    const memoryPath = buildSoulMemoryPath(item.id);
    expect(parseSoulMemoryPath(memoryPath)).toBe(item.id);
    const loaded = getSoulMemoryItem({ agentId: "main", itemId: item.id });
    expect(loaded?.content).toBe("remember this");
  });

  it("ingests recent facts into P3 and archive together", () => {
    const ingested = ingestSoulMemoryEvent({
      agentId: "main",
      scopeType: "session",
      scopeId: "agent:main:telegram:dm:user1",
      archiveScopeType: "agent",
      archiveScopeId: "main",
      kind: "user_turn",
      content: "We finished the release rollback and verified health checks.",
      summary: "User reported the release rollback was completed successfully.",
      recordKind: "fact",
      source: "runtime_event",
    });
    expect(ingested).not.toBeNull();
    if (!ingested) {
      throw new Error("ingested event missing");
    }

    expect(ingested.activeItem.tier).toBe("P3");
    expect(ingested.archiveEvent).toBeDefined();
    expect(countSoulArchiveEvents({ agentId: "main" })).toBe(1);

    const archiveEvent = ingested.archiveEvent!;
    const archive = getSoulArchiveEvent({
      agentId: "main",
      eventId: archiveEvent.id,
    });
    expect(archive?.summary).toContain("rollback");

    const archiveResults = querySoulArchive({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "rollback completed successfully",
      topK: 3,
      minScore: 0,
    });
    expect(archiveResults[0]?.id).toBe(archiveEvent.id);

    const archivePath = buildSoulArchivePath(archiveEvent.id);
    expect(parseSoulArchivePath(archivePath)).toBe(archiveEvent.id);
  });
});

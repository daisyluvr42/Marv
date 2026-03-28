import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { writeSoulMemory } from "../storage/soul-memory-store.js";
import { addTaskDecisionBookmark } from "./bookmark.js";
import { buildTaskContextPrelude, buildTaskContextWindow } from "./context-builder.js";
import { setTaskContextRollingSummary } from "./state.js";
import { createTaskContext, appendTaskContextEntry } from "./store.js";

let stateDir = "";
let previousStateDir: string | undefined;

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-task-builder-"));
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

describe("task context builder", () => {
  it("assembles layered context with summary, decisions, relevant memory, and recent turns", () => {
    createTaskContext({
      agentId: "main",
      taskId: "builder-task",
      title: "Builder Task",
      nowMs: 1000,
      scopeId: "task:main:builder-task",
    });
    setTaskContextRollingSummary({
      agentId: "main",
      taskId: "builder-task",
      summary: "Summarized project background and accepted constraints.",
      updatedAt: 1100,
    });
    addTaskDecisionBookmark({
      agentId: "main",
      taskId: "builder-task",
      content: "Must preserve backward compatibility with old session format.",
      createdAt: 1200,
    });
    // All items are palace tier now. Memory is included based on
    // relevance scoring from querySoulMemoryMulti, not tier-based filtering.
    writeSoulMemory({
      agentId: "main",
      scopeType: "task",
      scopeId: "task:main:builder-task",
      kind: "preference",
      content:
        "Continue implementation of the migration flow and include tests for behavior changes.",
      source: "core_preference",
    });

    for (let i = 0; i < 8; i += 1) {
      appendTaskContextEntry({
        agentId: "main",
        taskId: "builder-task",
        role: i % 2 === 0 ? "user" : "assistant",
        content: `Turn ${i} with details for context reconstruction.`,
        tokenCount: 220,
        createdAt: 1300 + i,
      });
    }

    const built = buildTaskContextWindow({
      agentId: "main",
      taskId: "builder-task",
      currentQuery: "Continue implementation of the migration flow and include tests.",
      toolContext: "Tools available: edit, test",
      layerBudgets: {
        recentEntries: 30,
      },
    });

    expect(built.layers.rollingSummary).toContain("Summarized project background");
    expect(built.layers.keyDecisions[0]).toContain("backward compatibility");
    // p0Memory layer now contains all relevance-scored items (all palace tier), not just P0-tier items
    expect(built.layers.p0Memory.some((line) => line.includes("migration flow"))).toBe(true);
    expect(built.layers.recentEntries.length).toBeGreaterThan(0);
    expect(built.layers.recentEntries.length).toBeLessThan(8);
    expect(built.messages[built.messages.length - 1]?.role).toBe("user");
    expect(built.tokenUsage.total).toBeGreaterThan(0);
  });

  it("builds a compact prelude string from retained layers", () => {
    createTaskContext({
      agentId: "main",
      taskId: "builder-prelude",
      title: "Prelude Task",
      nowMs: 1000,
    });
    setTaskContextRollingSummary({
      agentId: "main",
      taskId: "builder-prelude",
      summary: "High-level goal summary.",
      updatedAt: 1001,
    });
    addTaskDecisionBookmark({
      agentId: "main",
      taskId: "builder-prelude",
      content: "Decision: use isolated db per task.",
      createdAt: 1002,
    });

    const prelude = buildTaskContextPrelude({
      agentId: "main",
      taskId: "builder-prelude",
      currentQuery: "Continue task",
    });

    expect(prelude).toContain("Task Summary");
    expect(prelude).toContain("Key Decisions");
  });
});

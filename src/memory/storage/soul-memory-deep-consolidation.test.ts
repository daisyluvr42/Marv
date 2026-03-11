import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const inferLocalMock = vi.fn();

vi.mock("./local-llm-client.js", () => ({
  inferLocal: (...args: unknown[]) => inferLocalMock(...args),
  resolveLocalLlmConfig: () => ({
    api: "ollama",
    baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5:3b",
    timeoutMs: 30000,
    headers: {},
  }),
}));

import {
  formatSoulMemoryDeepConsolidationSummary,
  runSoulMemoryDeepConsolidation,
} from "./soul-memory-deep-consolidation.js";
import { resolveSoulMemoryDbPath, writeSoulMemory } from "./soul-memory-store.js";
import { requireNodeSqlite } from "./sqlite.js";

let stateDir = "";
let previousStateDir: string | undefined;

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-deep-consolidation-"));
  previousStateDir = process.env.MARV_STATE_DIR;
  process.env.MARV_STATE_DIR = stateDir;
  inferLocalMock.mockReset();
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

describe("runSoulMemoryDeepConsolidation", () => {
  it("generates summaries, conflict judgments, and cross-scope insights", async () => {
    writeSoulMemory({
      agentId: "main",
      scopeType: "project",
      scopeId: "deployments",
      kind: "habit",
      content: "After each deploy, send a short status update so the team knows what changed.",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "project",
      scopeId: "deployments",
      kind: "habit",
      content: "After every deployment, send a short status update so the team knows what changed.",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "project",
      scopeId: "deployments",
      kind: "habit",
      content: "Send a short post-deploy status update so the team knows what changed.",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "project",
      scopeId: "beverages",
      kind: "preference",
      content: "Coffee is better than tea for focus.",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "project",
      scopeId: "beverages",
      kind: "preference",
      content: "Tea is better than coffee for focus.",
      source: "manual_log",
    });
    writeSoulMemory({
      agentId: "main",
      scopeType: "project",
      scopeId: "health",
      kind: "principle",
      content: "Short feedback loops help health habits stick.",
      source: "manual_log",
    });

    inferLocalMock.mockImplementation(async (params: unknown) => {
      const input =
        typeof params === "object" && params
          ? (params as { system?: string; prompt?: string })
          : {};
      const system = input.system ?? "";
      const prompt = input.prompt ?? "";
      if (system.includes("cluster of related memories")) {
        return { ok: true, text: "Quick post-deploy updates reduce coordination confusion." };
      }
      if (system.includes("decide whether they genuinely conflict")) {
        if (
          prompt.includes("Coffee is better than tea for focus.") &&
          prompt.includes("Tea is better than coffee for focus.")
        ) {
          return { ok: true, text: "Opposite preference ranking for the same outcome." };
        }
        return { ok: true, text: "NO_CONFLICT" };
      }
      if (system.includes("cross-cutting pattern")) {
        return {
          ok: true,
          text: "Across work and health contexts, short feedback loops improve outcomes.",
        };
      }
      return { ok: false, error: "unexpected prompt" };
    });

    const report = await runSoulMemoryDeepConsolidation({
      cfg: {
        memory: {
          soul: {
            deepConsolidation: {
              enabled: true,
              model: {
                provider: "ollama",
                api: "ollama",
                model: "qwen2.5:3b",
              },
            },
          },
        },
      },
    });

    expect(report.totals.llmConsolidated).toBe(1);
    expect(report.totals.llmConflictsDetected).toBe(1);
    expect(report.totals.crossScopeReflections).toBe(1);
    expect(formatSoulMemoryDeepConsolidationSummary(report)).toContain("consolidated=1");

    const db = openDb("main");
    try {
      const insightRow = db
        .prepare(
          "SELECT content FROM memory_items WHERE scope_type = 'global' AND scope_id = 'cross-scope' AND kind = 'insight' LIMIT 1",
        )
        .get() as { content?: string } | undefined;
      expect(insightRow?.content).toContain("short feedback loops");

      const conflictRow = db
        .prepare("SELECT conflict_reason FROM memory_conflicts ORDER BY detected_at DESC LIMIT 1")
        .get() as { conflict_reason?: string } | undefined;
      expect(conflictRow?.conflict_reason).toContain("Opposite preference ranking");
    } finally {
      db.close();
    }
  });

  it("returns zero totals when there are no eligible candidates", async () => {
    writeSoulMemory({
      agentId: "main",
      scopeType: "project",
      scopeId: "solo",
      kind: "note",
      content: "Only one memory exists here.",
      source: "manual_log",
    });

    const report = await runSoulMemoryDeepConsolidation({
      cfg: {
        memory: {
          soul: {
            deepConsolidation: {
              enabled: true,
            },
          },
        },
      },
    });

    expect(report.totals.llmConsolidated).toBe(0);
    expect(report.totals.llmConflictsDetected).toBe(0);
    expect(report.totals.crossScopeReflections).toBe(0);
    expect(inferLocalMock).not.toHaveBeenCalled();
  });
});

function openDb(agentId: string) {
  const { DatabaseSync } = requireNodeSqlite();
  return new DatabaseSync(resolveSoulMemoryDbPath(agentId));
}

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
    model: "test-model",
    timeoutMs: 30000,
    headers: {},
  }),
}));

import {
  formatSoulMemoryDeepConsolidationSummary,
  runSoulMemoryDeepConsolidation,
} from "./soul-memory-deep-consolidation.js";
import { writeSoulMemory } from "./soul-memory-store.js";

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
  it("skips old 3-stage process (replaced by EXPERIENCE.md calibration)", async () => {
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

    // Mock inferLocal for the weeklyCalibration path (experience calibration)
    inferLocalMock.mockImplementation(async () => {
      return { ok: false, error: "no model available in test" };
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

    // Old 3-stage process is replaced — all LLM-based totals should be 0
    expect(report.totals.llmConsolidated).toBe(0);
    expect(report.totals.llmConflictsDetected).toBe(0);
    expect(report.totals.crossScopeReflections).toBe(0);
    expect(formatSoulMemoryDeepConsolidationSummary(report)).toContain("consolidated=0");

    // All three old stages should be reported as skipped/replaced
    const agent = report.agents[0];
    expect(agent).toBeDefined();
    expect(agent.skippedStages).toContain(
      "clusterSummarization-replaced-by-experience-calibration",
    );
    expect(agent.skippedStages).toContain("conflictJudgment-replaced-by-experience-calibration");
    expect(agent.skippedStages).toContain(
      "crossScopeReflection-replaced-by-experience-calibration",
    );
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

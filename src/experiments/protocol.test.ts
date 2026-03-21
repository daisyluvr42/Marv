import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ExperimentSpec, ExperimentState } from "./types.js";

// Mock the state dir so experiments go to a temp directory
const testState = vi.hoisted(() => ({
  stateDir: "",
}));

vi.mock("../core/config/paths.js", () => ({
  resolveStateDir: () => testState.stateDir,
}));

import { runExperiment, summarizeExperiment } from "./protocol.js";
import { renderExperimentLog } from "./results.js";
import { readExperimentStore } from "./store.js";

let stateDir: string;

function agentId(): string {
  return `test-${randomUUID().slice(0, 8)}`;
}

beforeEach(async () => {
  stateDir = path.join(os.tmpdir(), `marv-experiment-test-${randomUUID()}`);
  testState.stateDir = stateDir;
  await fs.mkdir(stateDir, { recursive: true });
});

afterEach(async () => {
  await fs.rm(stateDir, { recursive: true, force: true });
});

// ── Helper: create a spec with a simple evaluator ───────────────────

function makeSpec(overrides: Partial<ExperimentSpec> = {}): ExperimentSpec {
  return {
    id: `exp_${randomUUID().slice(0, 8)}`,
    name: "test experiment",
    evaluators: [
      {
        id: "metric",
        name: "test metric",
        measureCommand: 'echo "42"',
        metricParser: "first_number",
        direction: "higher_is_better",
      },
    ],
    objective: "Improve the test metric",
    constraints: {},
    maxIterations: 3,
    checkpoint: { strategy: "none" },
    ...overrides,
  };
}

// ── Tests ───────────────────────────────────────────────────────────

describe("runExperiment", () => {
  it("runs a basic experiment loop with improving metrics", async () => {
    const id = agentId();
    let callCount = 0;

    const spec = makeSpec({
      evaluators: [
        {
          id: "counter",
          name: "counter",
          // Each call returns an increasing number via a temp file
          measureCommand: "echo 42",
          metricParser: "first_number",
          direction: "higher_is_better",
        },
      ],
      maxIterations: 3,
    });

    // Track metric values: baseline=42, then simulate improvements
    const metricValues = [42, 45, 48, 50]; // baseline, iter0, iter1, iter2
    let _evalCall = 0;

    // Override the evaluator to return predictable values
    const originalSpec = { ...spec };
    // We'll use a temp file approach to simulate changing metrics
    const tmpFile = path.join(stateDir, "metric.txt");
    await fs.writeFile(tmpFile, "42");

    const specWithFile: ExperimentSpec = {
      ...originalSpec,
      evaluators: [
        {
          id: "counter",
          name: "counter",
          measureCommand: `cat "${tmpFile}"`,
          metricParser: "first_number",
          direction: "higher_is_better",
        },
      ],
    };

    const result = await runExperiment({
      spec: specWithFile,
      agentId: id,
      cwd: stateDir,
      runMutation: async ({ iteration }) => {
        // Simulate improvement by updating the metric file
        const nextValue = metricValues[iteration + 1] ?? metricValues[metricValues.length - 1];
        await fs.writeFile(tmpFile, String(nextValue));
        callCount++;
        return { summary: `Improved metric to ${nextValue}`, tokensUsed: 100 };
      },
    });

    expect(result.status).toBe("completed");
    expect(result.iterations.length).toBe(3);
    expect(callCount).toBe(3);
    // All iterations should be "improved" since metric keeps going up
    expect(result.iterations.every((i) => i.verdict === "improved")).toBe(true);
    expect(result.totalTokensUsed).toBe(300);
    expect(result.bestIteration).toBe(2);

    // Verify persisted in store
    const store = await readExperimentStore(id);
    expect(store.experiments.length).toBe(1);
    expect(store.experiments[0].spec.id).toBe(specWithFile.id);
  });

  it("rolls back on regression (no_change with no checkpoint)", async () => {
    const id = agentId();
    const tmpFile = path.join(stateDir, "metric.txt");
    await fs.writeFile(tmpFile, "42");

    const spec = makeSpec({
      evaluators: [
        {
          id: "val",
          name: "value",
          measureCommand: `cat "${tmpFile}"`,
          metricParser: "first_number",
          direction: "higher_is_better",
        },
      ],
      maxIterations: 2,
    });

    const result = await runExperiment({
      spec,
      agentId: id,
      cwd: stateDir,
      runMutation: async () => {
        // Don't change the metric — should result in no_change
        return { summary: "No real change", tokensUsed: 50 };
      },
    });

    expect(result.status).toBe("completed");
    expect(result.iterations.every((i) => i.verdict === "no_change")).toBe(true);
    expect(result.stopReason).toBe("max iterations reached");
  });

  it("stops early when threshold is met", async () => {
    const id = agentId();
    const tmpFile = path.join(stateDir, "metric.txt");
    await fs.writeFile(tmpFile, "42");

    const spec = makeSpec({
      evaluators: [
        {
          id: "val",
          name: "value",
          measureCommand: `cat "${tmpFile}"`,
          metricParser: "first_number",
          direction: "higher_is_better",
          threshold: 50,
        },
      ],
      maxIterations: 10,
    });

    let iterCount = 0;
    const result = await runExperiment({
      spec,
      agentId: id,
      cwd: stateDir,
      runMutation: async () => {
        iterCount++;
        // Jump to 55 on first iteration — above threshold
        await fs.writeFile(tmpFile, "55");
        return { summary: "Big improvement", tokensUsed: 200 };
      },
    });

    expect(result.status).toBe("completed");
    expect(result.stopReason).toBe("threshold met");
    expect(iterCount).toBe(1); // Should stop after first iteration
    expect(result.iterations[0].verdict).toBe("threshold_met");
  });

  it("fails when baseline measurement fails", async () => {
    const id = agentId();

    const spec = makeSpec({
      evaluators: [
        {
          id: "bad",
          name: "bad",
          measureCommand: "exit 1",
          metricParser: "first_number",
          direction: "higher_is_better",
          timeoutSeconds: 5,
        },
      ],
    });

    const result = await runExperiment({
      spec,
      agentId: id,
      cwd: stateDir,
      runMutation: async () => {
        throw new Error("Should not be called");
      },
    });

    expect(result.status).toBe("failed");
    expect(result.stopReason).toContain("Baseline measurement failed");
    expect(result.iterations.length).toBe(0);
  });

  it("respects token budget", async () => {
    const id = agentId();
    const tmpFile = path.join(stateDir, "metric.txt");
    await fs.writeFile(tmpFile, "42");

    const spec = makeSpec({
      evaluators: [
        {
          id: "val",
          name: "value",
          measureCommand: `cat "${tmpFile}"`,
          metricParser: "first_number",
          direction: "higher_is_better",
        },
      ],
      maxIterations: 100,
      constraints: { tokenBudget: 250 },
    });

    let iterCount = 0;
    const result = await runExperiment({
      spec,
      agentId: id,
      cwd: stateDir,
      runMutation: async () => {
        iterCount++;
        const newVal = 42 + iterCount;
        await fs.writeFile(tmpFile, String(newVal));
        return { summary: `iter ${iterCount}`, tokensUsed: 100 };
      },
    });

    // 100 tokens per iter, budget=250 → should complete 2 iters, then hit budget on 3rd check
    expect(result.status).toBe("completed");
    expect(result.stopReason).toBe("token budget exhausted");
    expect(iterCount).toBeLessThanOrEqual(3);
  });

  it("handles mutation errors gracefully", async () => {
    const id = agentId();
    const tmpFile = path.join(stateDir, "metric.txt");
    await fs.writeFile(tmpFile, "42");

    const spec = makeSpec({
      evaluators: [
        {
          id: "val",
          name: "value",
          measureCommand: `cat "${tmpFile}"`,
          metricParser: "first_number",
          direction: "higher_is_better",
        },
      ],
      maxIterations: 2,
    });

    const result = await runExperiment({
      spec,
      agentId: id,
      cwd: stateDir,
      runMutation: async () => {
        throw new Error("Agent crashed");
      },
    });

    expect(result.status).toBe("completed");
    expect(result.iterations.every((i) => i.verdict === "error")).toBe(true);
    expect(result.iterations[0].agentSummary).toContain("Agent crashed");
  });

  it("calls hooks at the right times", async () => {
    const id = agentId();
    const tmpFile = path.join(stateDir, "metric.txt");
    await fs.writeFile(tmpFile, "42");

    const spec = makeSpec({
      evaluators: [
        {
          id: "val",
          name: "value",
          measureCommand: `cat "${tmpFile}"`,
          metricParser: "first_number",
          direction: "higher_is_better",
        },
      ],
      maxIterations: 2,
    });

    const hookCalls: string[] = [];

    const _result = await runExperiment({
      spec,
      agentId: id,
      cwd: stateDir,
      runMutation: async ({ iteration }) => {
        await fs.writeFile(tmpFile, String(43 + iteration));
        return { summary: "ok", tokensUsed: 10 };
      },
      hooks: {
        beforeIteration: async (_state, i) => {
          hookCalls.push(`before-${i}`);
          return true;
        },
        afterIteration: async (_state, iter) => {
          hookCalls.push(`after-${iter.index}`);
        },
        onComplete: async () => {
          hookCalls.push("complete");
        },
      },
    });

    expect(hookCalls).toEqual(["before-0", "after-0", "before-1", "after-1", "complete"]);
  });

  it("stops when beforeIteration hook returns false", async () => {
    const id = agentId();
    const tmpFile = path.join(stateDir, "metric.txt");
    await fs.writeFile(tmpFile, "42");

    const spec = makeSpec({
      evaluators: [
        {
          id: "val",
          name: "value",
          measureCommand: `cat "${tmpFile}"`,
          metricParser: "first_number",
          direction: "higher_is_better",
        },
      ],
      maxIterations: 10,
    });

    const result = await runExperiment({
      spec,
      agentId: id,
      cwd: stateDir,
      runMutation: async () => ({ summary: "ok", tokensUsed: 10 }),
      hooks: {
        beforeIteration: async (_state, i) => i < 1, // Stop after first
      },
    });

    expect(result.status).toBe("stopped");
    expect(result.stopReason).toBe("stopped by hook");
    expect(result.iterations.length).toBe(1);
  });

  it("yields to user when shouldYield returns true", async () => {
    const id = agentId();
    const tmpFile = path.join(stateDir, "metric.txt");
    await fs.writeFile(tmpFile, "42");

    const spec = makeSpec({ maxIterations: 10 });
    spec.evaluators[0].measureCommand = `cat "${tmpFile}"`;

    const result = await runExperiment({
      spec,
      agentId: id,
      cwd: stateDir,
      runMutation: async () => ({ summary: "ok", tokensUsed: 10 }),
      shouldYield: () => true, // Always yield
    });

    expect(result.status).toBe("stopped");
    expect(result.stopReason).toBe("yielded to user");
    expect(result.iterations.length).toBe(0);
  });
});

// ── Multi-evaluator ─────────────────────────────────────────────────

describe("runExperiment (multi-evaluator)", () => {
  it("keeps only when all evaluators pass", async () => {
    const id = agentId();
    const metricFile = path.join(stateDir, "coverage.txt");
    const speedFile = path.join(stateDir, "speed.txt");
    await fs.writeFile(metricFile, "70");
    await fs.writeFile(speedFile, "100");

    const spec = makeSpec({
      evaluators: [
        {
          id: "coverage",
          name: "coverage",
          measureCommand: `cat "${metricFile}"`,
          metricParser: "first_number",
          direction: "higher_is_better",
        },
        {
          id: "speed",
          name: "speed",
          measureCommand: `cat "${speedFile}"`,
          metricParser: "first_number",
          direction: "lower_is_better",
        },
      ],
      maxIterations: 2,
    });

    let iter = 0;
    const result = await runExperiment({
      spec,
      agentId: id,
      cwd: stateDir,
      runMutation: async () => {
        iter++;
        if (iter === 1) {
          // Improve coverage but regress speed
          await fs.writeFile(metricFile, "75");
          await fs.writeFile(speedFile, "110");
        } else {
          // Improve both
          await fs.writeFile(metricFile, "75");
          await fs.writeFile(speedFile, "90");
        }
        return { summary: `iter ${iter}`, tokensUsed: 50 };
      },
    });

    // Iter 0: coverage up, speed down → regressed (speed regressed)
    expect(result.iterations[0].verdict).toBe("regressed");
    // Iter 1: both improved → improved
    expect(result.iterations[1].verdict).toBe("improved");
  });
});

// ── renderExperimentLog ─────────────────────────────────────────────

describe("renderExperimentLog", () => {
  it("renders a readable markdown log", () => {
    const state: ExperimentState = {
      spec: makeSpec(),
      status: "completed",
      iterations: [
        {
          index: 0,
          baseline: [
            { evaluatorId: "metric", value: 42, raw: "42", measuredAt: 0, durationMs: 10 },
          ],
          candidate: [
            { evaluatorId: "metric", value: 45, raw: "45", measuredAt: 0, durationMs: 10 },
          ],
          verdict: "improved",
          agentSummary: "Increased batch size",
          tokensUsed: 100,
          durationMs: 5000,
          startedAt: 0,
        },
      ],
      bestResult: [{ evaluatorId: "metric", value: 45, raw: "45", measuredAt: 0, durationMs: 10 }],
      bestIteration: 0,
      totalTokensUsed: 100,
      startedAt: 0,
      completedAt: 5000,
      stopReason: "max iterations reached",
    };

    const log = renderExperimentLog(state);
    expect(log).toContain("# Experiment: test experiment");
    expect(log).toContain("Increased batch size");
    expect(log).toContain("(kept)");
    expect(log).toContain("**metric:** 45");
    expect(log).toContain("1 kept");
  });
});

// ── summarizeExperiment ─────────────────────────────────────────────

describe("summarizeExperiment", () => {
  it("produces a concise summary", () => {
    const state: ExperimentState = {
      spec: makeSpec(),
      status: "completed",
      iterations: [
        {
          index: 0,
          baseline: [{ evaluatorId: "metric", value: 42, raw: "", measuredAt: 0, durationMs: 10 }],
          candidate: [{ evaluatorId: "metric", value: 45, raw: "", measuredAt: 0, durationMs: 10 }],
          verdict: "improved",
          tokensUsed: 100,
          durationMs: 5000,
          startedAt: 0,
        },
        {
          index: 1,
          baseline: [{ evaluatorId: "metric", value: 45, raw: "", measuredAt: 0, durationMs: 10 }],
          candidate: [{ evaluatorId: "metric", value: 44, raw: "", measuredAt: 0, durationMs: 10 }],
          verdict: "regressed",
          tokensUsed: 100,
          durationMs: 3000,
          startedAt: 0,
        },
      ],
      bestResult: [{ evaluatorId: "metric", value: 45, raw: "", measuredAt: 0, durationMs: 10 }],
      bestIteration: 0,
      totalTokensUsed: 200,
      startedAt: 0,
      completedAt: 8000,
    };

    const summary = summarizeExperiment(state);
    expect(summary).toContain("test experiment");
    expect(summary).toContain("1 kept");
    expect(summary).toContain("1 rolled back");
    expect(summary).toContain("200");
  });
});

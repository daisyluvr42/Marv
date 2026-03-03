import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { compressTaskContext, maybeCompressTaskContext } from "./compressor.js";
import { createTaskContext, listTaskContextEntries, appendTaskContextEntry } from "./store.js";

let stateDir = "";
let previousStateDir: string | undefined;

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-task-compressor-"));
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

describe("task context compressor", () => {
  it("compresses oldest turns into rolling summary when threshold is exceeded", async () => {
    createTaskContext({
      agentId: "main",
      taskId: "compress-target",
      title: "Compression Task",
      nowMs: 1000,
    });

    for (let i = 0; i < 12; i += 1) {
      appendTaskContextEntry({
        agentId: "main",
        taskId: "compress-target",
        role: i % 2 === 0 ? "user" : "assistant",
        content: `Turn ${i} content with enough words to consume tokens and force summarization.`,
        tokenCount: 900,
        createdAt: 1000 + i,
      });
    }

    const result = await compressTaskContext({
      agentId: "main",
      taskId: "compress-target",
      recentTokenThreshold: 2000,
      batchTokenTarget: 1800,
      keepRecentTurns: 3,
    });

    expect(result.compressed).toBe(true);
    expect(result.summarizedEntries).toBeGreaterThan(0);
    expect(result.summarizedTokens).toBeGreaterThan(0);
    expect(result.rollingSummary).toContain("Turn");

    const allEntries = listTaskContextEntries({
      agentId: "main",
      taskId: "compress-target",
      limit: 200,
    });
    const summarizedCount = allEntries.filter((entry) => entry.summarized).length;
    expect(summarizedCount).toBe(result.summarizedEntries);
  });

  it("skips compression when recent context is below threshold", async () => {
    createTaskContext({
      agentId: "main",
      taskId: "compress-skip",
      title: "Skip Task",
      nowMs: 1000,
    });
    appendTaskContextEntry({
      agentId: "main",
      taskId: "compress-skip",
      role: "user",
      content: "Short turn",
      tokenCount: 5,
      createdAt: 1001,
    });
    appendTaskContextEntry({
      agentId: "main",
      taskId: "compress-skip",
      role: "assistant",
      content: "Short reply",
      tokenCount: 5,
      createdAt: 1002,
    });

    const result = await maybeCompressTaskContext({
      agentId: "main",
      taskId: "compress-skip",
      thresholdTokens: 20,
    });

    expect(result.compressed).toBe(false);
    expect(result.reason).toBe("below-threshold");
  });
});

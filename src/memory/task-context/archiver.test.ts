import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { archiveTask } from "./archiver.js";
import { addTaskDecisionBookmark } from "./bookmark.js";
import { setTaskContextRollingSummary } from "./state.js";
import { createTaskContext, getTaskContext, appendTaskContextEntry } from "./store.js";

let stateDir = "";
let previousStateDir: string | undefined;

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-task-archive-"));
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

describe("task archiver", () => {
  it("exports context jsonl and markdown report then marks task archived", async () => {
    createTaskContext({
      agentId: "main",
      taskId: "archive-me",
      title: "Archive Me",
      nowMs: 1000,
    });
    appendTaskContextEntry({
      agentId: "main",
      taskId: "archive-me",
      role: "user",
      content: "Please implement feature X",
      tokenCount: 8,
      createdAt: 1100,
    });
    appendTaskContextEntry({
      agentId: "main",
      taskId: "archive-me",
      role: "assistant",
      content: "Implemented feature X with tests",
      tokenCount: 10,
      createdAt: 1200,
    });
    setTaskContextRollingSummary({
      agentId: "main",
      taskId: "archive-me",
      summary: "Feature X implementation completed.",
    });
    addTaskDecisionBookmark({
      agentId: "main",
      taskId: "archive-me",
      content: "Must keep backward compatibility.",
    });

    const archive = await archiveTask({
      agentId: "main",
      taskId: "archive-me",
      nowMs: 1300,
    });

    const jsonl = await fs.readFile(archive.contextJsonlPath, "utf-8");
    const report = await fs.readFile(archive.reportMarkdownPath, "utf-8");
    expect(jsonl).toContain('"role":"user"');
    expect(jsonl).toContain('"role":"assistant"');
    expect(report).toContain("Task Archive Report");
    expect(report).toContain("Feature X implementation completed.");
    expect(report).toContain("backward compatibility");

    const task = getTaskContext({ agentId: "main", taskId: "archive-me" });
    expect(task?.status).toBe("archived");
    expect(archive.entryCount).toBe(2);
    expect(archive.totalTokens).toBe(18);
  });
});

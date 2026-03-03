import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { TaskArchive } from "./archiver.js";
import { distillTaskContext } from "./distiller.js";

let tempDir = "";

beforeEach(async () => {
  tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-task-distill-"));
});

afterEach(async () => {
  if (tempDir) {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
});

function makeArchive(contextJsonlPath: string): TaskArchive {
  return {
    agentId: "main",
    taskId: "distill-task",
    archiveDir: path.dirname(contextJsonlPath),
    contextJsonlPath,
    reportMarkdownPath: path.join(path.dirname(contextJsonlPath), "report.md"),
    archivedAt: Date.now(),
    entryCount: 0,
    totalTokens: 0,
  };
}

describe("distillTaskContext", () => {
  it("extracts heuristic facts, preferences, lessons, and workflow skills", async () => {
    const contextPath = path.join(tempDir, "context.jsonl");
    await fs.writeFile(
      contextPath,
      [
        {
          sequence: 1,
          role: "user",
          content: "I prefer concise summaries and please always include tests.",
        },
        {
          sequence: 2,
          role: "assistant",
          content: "The system supports SQLite and requires Node 22.",
        },
        {
          sequence: 3,
          role: "assistant",
          content: "Lesson learned: next time validate schemas first.",
        },
        {
          sequence: 4,
          role: "assistant",
          content: "Run pnpm test before release.",
        },
        {
          sequence: 5,
          role: "assistant",
          content: "Create migration file and update tests.",
        },
        {
          sequence: 6,
          role: "tool",
          content: "Verify deployment and check logs.",
        },
      ]
        .map((entry) => JSON.stringify(entry))
        .join("\n"),
      "utf-8",
    );

    const distilled = await distillTaskContext({
      agentId: "main",
      taskId: "distill-task",
      archive: makeArchive(contextPath),
    });

    expect(distilled.preferences.length).toBeGreaterThan(0);
    expect(distilled.facts.length).toBeGreaterThan(0);
    expect(distilled.lessons.length).toBeGreaterThan(0);
    expect(distilled.skills.length).toBe(1);
    expect(distilled.skills[0]?.steps.length).toBeGreaterThanOrEqual(3);
  });

  it("deduplicates llm-distilled outputs", async () => {
    const contextPath = path.join(tempDir, "context.jsonl");
    await fs.writeFile(
      contextPath,
      `${JSON.stringify({ sequence: 1, role: "assistant", content: "noop" })}\n`,
      "utf-8",
    );

    const distilled = await distillTaskContext({
      agentId: "main",
      taskId: "distill-task",
      archive: makeArchive(contextPath),
      distillWithLlm: async () => ({
        facts: [
          { content: "Use strict typing.", kind: "fact", confidence: 0.6 },
          { content: "Use strict typing. ", kind: "fact", confidence: 0.7 },
        ],
        preferences: [
          { content: "Always include tests", confidence: 0.8 },
          { content: "Always include tests", confidence: 0.8 },
        ],
        lessons: [
          { content: "Validate before deploy", context: "a" },
          { content: "Validate before deploy", context: "b" },
        ],
        skills: [
          {
            name: "release-flow",
            description: "Release safely",
            steps: ["Run tests", "Run tests", "Deploy"],
          },
          {
            name: "release-flow",
            description: "Release safely",
            steps: ["Run tests", "Run tests", "Deploy"],
          },
        ],
      }),
    });

    expect(distilled.facts).toHaveLength(1);
    expect(distilled.preferences).toHaveLength(1);
    expect(distilled.lessons).toHaveLength(1);
    expect(distilled.skills).toHaveLength(1);
    expect(distilled.skills[0]?.steps).toEqual(["Run tests", "Deploy"]);
  });
});

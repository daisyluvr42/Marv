import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { resolveStateDir } from "../../core/config/paths.js";
import { listTaskDecisionBookmarks } from "./bookmark.js";
import { getTaskContextRollingSummary } from "./state.js";
import { getTaskContext, listTaskContextEntries, updateTaskContextStatus } from "./store.js";
import type { TaskContextEntry } from "./types.js";

export type TaskArchive = {
  agentId: string;
  taskId: string;
  archiveDir: string;
  contextJsonlPath: string;
  reportMarkdownPath: string;
  archivedAt: number;
  entryCount: number;
  totalTokens: number;
};

export async function archiveTask(params: {
  agentId: string;
  taskId: string;
  nowMs?: number;
  env?: NodeJS.ProcessEnv;
}): Promise<TaskArchive> {
  const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
  const task = getTaskContext({
    agentId: params.agentId,
    taskId: params.taskId,
    env: params.env,
  });
  if (!task) {
    throw new Error(`Task context not found: ${params.taskId}`);
  }

  const entries = listTaskContextEntries({
    agentId: params.agentId,
    taskId: params.taskId,
    limit: 50_000,
    env: params.env,
  });
  const baseArchiveDir = resolveTaskArchiveDir({
    agentId: params.agentId,
    taskId: params.taskId,
    env: params.env,
  });
  await fs.mkdir(baseArchiveDir, { recursive: true });

  const contextJsonlPath = path.join(baseArchiveDir, "context.jsonl");
  const reportMarkdownPath = path.join(baseArchiveDir, "report.md");

  await fs.writeFile(contextJsonlPath, buildArchiveJsonl(entries), "utf-8");
  await fs.writeFile(
    reportMarkdownPath,
    buildArchiveReport({
      taskTitle: task.title,
      taskId: task.taskId,
      agentId: task.agentId,
      scopeId: task.scopeId,
      createdAt: task.createdAt,
      archivedAt: nowMs,
      entries,
      rollingSummary:
        getTaskContextRollingSummary({
          agentId: params.agentId,
          taskId: params.taskId,
          env: params.env,
        }) ?? undefined,
      decisions: listTaskDecisionBookmarks({
        agentId: params.agentId,
        taskId: params.taskId,
        limit: 200,
        env: params.env,
      }).map((bookmark) => bookmark.content),
    }),
    "utf-8",
  );

  updateTaskContextStatus({
    agentId: params.agentId,
    taskId: params.taskId,
    status: "archived",
    updatedAt: nowMs,
    env: params.env,
  });

  return {
    agentId: params.agentId,
    taskId: params.taskId,
    archiveDir: baseArchiveDir,
    contextJsonlPath,
    reportMarkdownPath,
    archivedAt: nowMs,
    entryCount: entries.length,
    totalTokens: entries.reduce((sum, entry) => sum + Math.max(1, entry.tokenCount), 0),
  };
}

export function resolveTaskArchiveDir(params: {
  agentId: string;
  taskId: string;
  env?: NodeJS.ProcessEnv;
}): string {
  const stateDir = resolveStateDir(params.env ?? process.env, os.homedir);
  return path.join(stateDir, "archives", params.agentId, params.taskId);
}

function buildArchiveJsonl(entries: TaskContextEntry[]): string {
  return entries
    .map((entry) =>
      JSON.stringify({
        id: entry.id,
        taskId: entry.taskId,
        sequence: entry.sequence,
        role: entry.role,
        content: entry.content,
        contentHash: entry.contentHash,
        summary: entry.summary ?? null,
        tokenCount: entry.tokenCount,
        createdAt: entry.createdAt,
        metadata: entry.metadata ?? null,
        summarized: entry.summarized,
      }),
    )
    .join("\n");
}

function buildArchiveReport(params: {
  taskTitle: string;
  taskId: string;
  agentId: string;
  scopeId: string;
  createdAt: number;
  archivedAt: number;
  entries: TaskContextEntry[];
  rollingSummary?: string;
  decisions: string[];
}): string {
  const totalTokens = params.entries.reduce((sum, entry) => sum + Math.max(1, entry.tokenCount), 0);
  const timelineLines = params.entries.slice(0, 50).map((entry) => {
    const ts = new Date(entry.createdAt).toISOString();
    const text = entry.content.replace(/\s+/g, " ").trim();
    const short = text.length > 120 ? `${text.slice(0, 117)}...` : text;
    return `- [${ts}] (${entry.role}) ${short}`;
  });

  const decisions = params.decisions.length
    ? params.decisions.map((line) => `- ${line.trim()}`).join("\n")
    : "- (none)";
  const rollingSummary = params.rollingSummary?.trim() ? params.rollingSummary.trim() : "(none)";

  return [
    `# Task Archive Report`,
    ``,
    `## Metadata`,
    `- Task: ${params.taskTitle}`,
    `- Task ID: ${params.taskId}`,
    `- Agent: ${params.agentId}`,
    `- Scope: ${params.scopeId}`,
    `- Created At: ${new Date(params.createdAt).toISOString()}`,
    `- Archived At: ${new Date(params.archivedAt).toISOString()}`,
    `- Entries: ${params.entries.length}`,
    `- Total Tokens (estimated): ${totalTokens}`,
    ``,
    `## Rolling Summary`,
    rollingSummary,
    ``,
    `## Key Decisions`,
    decisions,
    ``,
    `## Timeline (first 50 entries)`,
    ...(timelineLines.length > 0 ? timelineLines : ["- (no entries)"]),
    ``,
  ].join("\n");
}

import { resolveDefaultAgentId } from "../agents/agent-scope.js";
import { loadConfig } from "../core/config/config.js";
import {
  archiveTask,
  distillTaskContext,
  getTaskContext,
  injectDistilledKnowledge,
  listTaskContextEntries,
  listTaskContextsForAgent,
  listTaskDecisionBookmarks,
  getTaskContextRollingSummary,
  updateTaskContextStatus,
  type TaskStatus,
} from "../memory/task-context/index.js";
import type { RuntimeEnv } from "../runtime.js";
export type { TaskStatus } from "../memory/task-context/index.js";

type TaskListCommandOptions = {
  agent?: string;
  status?: TaskStatus;
  limit?: number;
  json?: boolean;
};

type TaskShowCommandOptions = {
  agent?: string;
  taskId: string;
  entries?: number;
  json?: boolean;
};

type TaskArchiveCommandOptions = {
  agent?: string;
  taskId: string;
  archiveOnly?: boolean;
  json?: boolean;
};

function resolveAgentId(agent: string | undefined): string {
  const cfg = loadConfig();
  return agent?.trim() || resolveDefaultAgentId(cfg);
}

function normalizeLimit(raw: number | undefined, fallback: number, max: number): number {
  if (!Number.isFinite(raw)) {
    return fallback;
  }
  return Math.max(1, Math.min(max, Math.floor(raw as number)));
}

function formatTaskLine(task: {
  taskId: string;
  status: string;
  updatedAt: number;
  title: string;
  totalEntries: number;
  totalTokens: number;
}): string {
  const updatedAt = new Date(task.updatedAt).toISOString();
  return `${task.taskId} | ${task.status} | entries=${task.totalEntries} | tokens=${task.totalTokens} | updated=${updatedAt} | ${task.title}`;
}

export async function taskListCommand(
  opts: TaskListCommandOptions,
  runtime: RuntimeEnv,
): Promise<void> {
  const agentId = resolveAgentId(opts.agent);
  const tasks = listTaskContextsForAgent({
    agentId,
    status: opts.status,
    limit: normalizeLimit(opts.limit, 50, 1000),
  });

  if (opts.json) {
    runtime.log(
      JSON.stringify(
        {
          agentId,
          count: tasks.length,
          tasks,
        },
        null,
        2,
      ),
    );
    return;
  }

  runtime.log(`Agent: ${agentId}`);
  runtime.log(`Tasks: ${tasks.length}`);
  if (tasks.length === 0) {
    runtime.log("No tasks found.");
    return;
  }
  for (const task of tasks) {
    runtime.log(formatTaskLine(task));
  }
}

export async function taskShowCommand(
  opts: TaskShowCommandOptions,
  runtime: RuntimeEnv,
): Promise<void> {
  const agentId = resolveAgentId(opts.agent);
  const task = getTaskContext({
    agentId,
    taskId: opts.taskId,
  });
  if (!task) {
    runtime.error(`Task not found: ${opts.taskId}`);
    runtime.exit(1);
    return;
  }

  const entries = listTaskContextEntries({
    agentId,
    taskId: task.taskId,
    limit: normalizeLimit(opts.entries, 120, 2000),
  });
  const rollingSummary = getTaskContextRollingSummary({
    agentId,
    taskId: task.taskId,
  });
  const decisions = listTaskDecisionBookmarks({
    agentId,
    taskId: task.taskId,
    limit: 80,
  });

  if (opts.json) {
    runtime.log(
      JSON.stringify(
        {
          agentId,
          task,
          rollingSummary,
          decisions,
          entries,
        },
        null,
        2,
      ),
    );
    return;
  }

  runtime.log(`Task: ${task.taskId}`);
  runtime.log(`Title: ${task.title}`);
  runtime.log(`Status: ${task.status}`);
  runtime.log(`Scope: ${task.scopeId}`);
  runtime.log(`Entries: ${task.totalEntries} (showing ${entries.length})`);
  runtime.log(`Tokens: ${task.totalTokens}`);
  runtime.log(`Updated: ${new Date(task.updatedAt).toISOString()}`);
  if (rollingSummary?.trim()) {
    runtime.log("");
    runtime.log("Rolling Summary:");
    runtime.log(rollingSummary.trim());
  }
  if (decisions.length > 0) {
    runtime.log("");
    runtime.log("Key Decisions:");
    for (const decision of decisions) {
      runtime.log(`- ${decision.content}`);
    }
  }
  if (entries.length > 0) {
    runtime.log("");
    runtime.log("Recent Entries:");
    for (const entry of entries) {
      const ts = new Date(entry.createdAt).toISOString();
      const content = entry.content.replace(/\s+/g, " ").trim();
      const short = content.length > 140 ? `${content.slice(0, 137)}...` : content;
      runtime.log(`${entry.sequence}. [${ts}] (${entry.role}) ${short}`);
    }
  }
}

export async function taskArchiveCommand(
  opts: TaskArchiveCommandOptions,
  runtime: RuntimeEnv,
): Promise<void> {
  const agentId = resolveAgentId(opts.agent);
  const task = getTaskContext({
    agentId,
    taskId: opts.taskId,
  });
  if (!task) {
    runtime.error(`Task not found: ${opts.taskId}`);
    runtime.exit(1);
    return;
  }

  const nowMs = Date.now();
  if (task.status === "active" || task.status === "paused") {
    updateTaskContextStatus({
      agentId,
      taskId: task.taskId,
      status: "completed",
      updatedAt: nowMs,
      completedAt: nowMs,
    });
  }

  const archive = await archiveTask({
    agentId,
    taskId: task.taskId,
    nowMs,
  });

  let distilledSummary:
    | {
        facts: number;
        preferences: number;
        lessons: number;
        skills: number;
      }
    | undefined;
  let injectSummary:
    | {
        insertedFacts: number;
        insertedPreferences: number;
        insertedLessons: number;
        insertedSkillsAsLessons: number;
        skippedAsDuplicate: number;
      }
    | undefined;

  if (!opts.archiveOnly) {
    const distilled = await distillTaskContext({
      agentId,
      taskId: task.taskId,
      archive,
    });
    distilledSummary = {
      facts: distilled.facts.length,
      preferences: distilled.preferences.length,
      lessons: distilled.lessons.length,
      skills: distilled.skills.length,
    };
    const injected = injectDistilledKnowledge({
      agentId,
      taskScopeId: task.scopeId,
      distilled,
    });
    injectSummary = {
      insertedFacts: injected.insertedFacts,
      insertedPreferences: injected.insertedPreferences,
      insertedLessons: injected.insertedLessons,
      insertedSkillsAsLessons: injected.insertedSkillsAsLessons,
      skippedAsDuplicate: injected.skippedAsDuplicate,
    };
  }

  if (opts.json) {
    runtime.log(
      JSON.stringify(
        {
          agentId,
          taskId: task.taskId,
          archive,
          distilled: distilledSummary,
          injected: injectSummary,
        },
        null,
        2,
      ),
    );
    return;
  }

  runtime.log(`Archived task ${task.taskId}`);
  runtime.log(`- JSONL: ${archive.contextJsonlPath}`);
  runtime.log(`- Report: ${archive.reportMarkdownPath}`);
  runtime.log(`- Entries: ${archive.entryCount}`);
  runtime.log(`- Tokens: ${archive.totalTokens}`);
  if (distilledSummary) {
    runtime.log(
      `- Distilled: facts=${distilledSummary.facts}, preferences=${distilledSummary.preferences}, lessons=${distilledSummary.lessons}, skills=${distilledSummary.skills}`,
    );
  }
  if (injectSummary) {
    runtime.log(
      `- Injected: facts=${injectSummary.insertedFacts}, preferences=${injectSummary.insertedPreferences}, lessons=${injectSummary.insertedLessons}, skillsAsLessons=${injectSummary.insertedSkillsAsLessons}, skippedDuplicates=${injectSummary.skippedAsDuplicate}`,
    );
  }
}

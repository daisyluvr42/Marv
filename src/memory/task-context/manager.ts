import crypto from "node:crypto";
import { normalizeAgentId } from "../../routing/session-key.js";
import {
  appendTaskContextEntry,
  createTaskContext,
  getTaskContext,
  listTaskContextEntries,
  listTaskContextsForAgent,
  normalizeTaskId,
  resolveTaskContextDbPath,
  updateTaskContextStatus,
  type AppendTaskContextEntryParams,
  type ListTaskContextEntriesParams,
} from "./store.js";
import type { TaskContext, TaskContextEntry } from "./types.js";

export type StartTaskParams = {
  agentId: string;
  taskId?: string;
  title: string;
  parentTaskId?: string;
  scopeId?: string;
  nowMs?: number;
  env?: NodeJS.ProcessEnv;
};

export type ListTasksParams = {
  agentId: string;
  status?: "active" | "paused" | "completed" | "archived";
  limit?: number;
  env?: NodeJS.ProcessEnv;
};

export class TaskContextManager {
  startTask(params: StartTaskParams): TaskContext {
    const agentId = normalizeAgentId(params.agentId);
    const title = params.title.trim();
    if (!title) {
      throw new Error("Task title required");
    }
    const nowMs = Number.isFinite(params.nowMs) ? Math.floor(params.nowMs as number) : Date.now();
    const taskId = params.taskId?.trim()
      ? normalizeTaskId(params.taskId)
      : this.generateTaskId(agentId, title, nowMs, params.env);
    return createTaskContext({
      agentId,
      taskId,
      title,
      parentTaskId: params.parentTaskId,
      scopeId: params.scopeId,
      nowMs,
      env: params.env,
    });
  }

  getTask(params: {
    agentId: string;
    taskId: string;
    env?: NodeJS.ProcessEnv;
  }): TaskContext | null {
    return getTaskContext(params);
  }

  listTasks(params: ListTasksParams): TaskContext[] {
    return listTaskContextsForAgent(params);
  }

  appendEntry(params: AppendTaskContextEntryParams): TaskContextEntry | null {
    return appendTaskContextEntry(params);
  }

  listEntries(params: ListTaskContextEntriesParams): TaskContextEntry[] {
    return listTaskContextEntries(params);
  }

  pauseTask(params: {
    agentId: string;
    taskId: string;
    nowMs?: number;
    env?: NodeJS.ProcessEnv;
  }): TaskContext | null {
    return updateTaskContextStatus({
      agentId: params.agentId,
      taskId: params.taskId,
      status: "paused",
      updatedAt: params.nowMs,
      env: params.env,
    });
  }

  resumeTask(params: {
    agentId: string;
    taskId: string;
    nowMs?: number;
    env?: NodeJS.ProcessEnv;
  }): TaskContext | null {
    return updateTaskContextStatus({
      agentId: params.agentId,
      taskId: params.taskId,
      status: "active",
      updatedAt: params.nowMs,
      env: params.env,
    });
  }

  completeTask(params: {
    agentId: string;
    taskId: string;
    nowMs?: number;
    env?: NodeJS.ProcessEnv;
  }): TaskContext | null {
    return updateTaskContextStatus({
      agentId: params.agentId,
      taskId: params.taskId,
      status: "completed",
      updatedAt: params.nowMs,
      completedAt: params.nowMs,
      env: params.env,
    });
  }

  archiveTask(params: {
    agentId: string;
    taskId: string;
    nowMs?: number;
    env?: NodeJS.ProcessEnv;
  }): TaskContext | null {
    return updateTaskContextStatus({
      agentId: params.agentId,
      taskId: params.taskId,
      status: "archived",
      updatedAt: params.nowMs,
      env: params.env,
    });
  }

  resolveDbPath(params: { agentId: string; taskId: string; env?: NodeJS.ProcessEnv }): string {
    return resolveTaskContextDbPath(params);
  }

  private generateTaskId(
    agentId: string,
    title: string,
    nowMs: number,
    env?: NodeJS.ProcessEnv,
  ): string {
    const titleBase = normalizeTaskId(
      title
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+/, "")
        .replace(/-+$/, "")
        .slice(0, 48) || "task",
    );
    const timePart = nowMs.toString(36);
    for (let attempt = 0; attempt < 8; attempt += 1) {
      const randomPart = crypto.randomUUID().replace(/-/g, "").slice(0, 6);
      const candidate = normalizeTaskId(`${titleBase}-${timePart}-${randomPart}`);
      if (!getTaskContext({ agentId, taskId: candidate, env })) {
        return candidate;
      }
    }
    return normalizeTaskId(`${titleBase}-${Date.now().toString(36)}`);
  }
}

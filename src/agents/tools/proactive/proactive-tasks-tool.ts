import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../../core/config/config.js";
import { addGoal, listGoals, removeGoal, updateGoal } from "../../../proactive/goals.js";
import {
  completeTask,
  enqueueTaskIdempotent,
  listTasks,
  pauseTask,
} from "../../../proactive/task-queue.js";
import { resolveSessionAgentId } from "../../agent-scope.js";
import { stringEnum } from "../../schema/typebox.js";
import type { AnyAgentTool } from "../common.js";
import { jsonResult, readStringParam } from "../common.js";

const ACTIONS = [
  "add_goal",
  "list_goals",
  "update_goal",
  "remove_goal",
  "add_task",
  "list_tasks",
  "complete_task",
  "pause_task",
] as const;

const ProactiveTasksSchema = Type.Object({
  action: stringEnum(ACTIONS),
  // Goal fields
  goal_id: Type.Optional(Type.String()),
  // Task fields
  task_id: Type.Optional(Type.String()),
  fingerprint: Type.Optional(Type.String()),
  // Shared fields
  title: Type.Optional(Type.String()),
  description: Type.Optional(Type.String()),
  priority: Type.Optional(Type.String()),
  status: Type.Optional(Type.String()),
  // Result on completion
  result: Type.Optional(Type.String()),
  // Filters
  filter_status: Type.Optional(Type.String()),
});

export function createProactiveTasksTool(options: {
  config?: MarvConfig;
  agentSessionKey?: string;
  enabled?: boolean;
}): AnyAgentTool | null {
  if (!options.enabled) {
    return null;
  }
  const agentId = resolveSessionAgentId({
    sessionKey: options.agentSessionKey,
    config: options.config,
  });
  return {
    label: "Proactive Tasks",
    name: "proactive_tasks",
    description:
      "Manage proactive goals and tasks. Goals are high-level objectives; tasks are concrete units of work derived from goals.",
    parameters: ProactiveTasksSchema,
    execute: async (_toolCallId, params) => {
      const action = readStringParam(params, "action", { required: true });

      // ── Goal actions ─────────────────────────────────────────────
      if (action === "add_goal") {
        const title = readStringParam(params, "title", { required: true });
        const description = readStringParam(params, "description", { required: true });
        const priority = readStringParam(params, "priority") as
          | "high"
          | "normal"
          | "low"
          | undefined;
        const goal = await addGoal(agentId, { title, description, priority });
        return jsonResult({ created: goal });
      }

      if (action === "list_goals") {
        const filterStatus = readStringParam(params, "filter_status") as
          | "active"
          | "paused"
          | "completed"
          | undefined;
        const goals = await listGoals(agentId, filterStatus ? { status: filterStatus } : undefined);
        return jsonResult({ goals });
      }

      if (action === "update_goal") {
        const goalId = readStringParam(params, "goal_id", { required: true });
        const patch: Record<string, string> = {};
        const title = readStringParam(params, "title");
        const description = readStringParam(params, "description");
        const priority = readStringParam(params, "priority");
        const status = readStringParam(params, "status");
        if (title) {
          patch.title = title;
        }
        if (description) {
          patch.description = description;
        }
        if (priority) {
          patch.priority = priority;
        }
        if (status) {
          patch.status = status;
        }
        const updated = await updateGoal(agentId, goalId, patch);
        if (!updated) {
          return jsonResult({ error: "goal not found" });
        }
        return jsonResult({ updated });
      }

      if (action === "remove_goal") {
        const goalId = readStringParam(params, "goal_id", { required: true });
        const removed = await removeGoal(agentId, goalId);
        return jsonResult({ removed });
      }

      // ── Task actions ─────────────────────────────────────────────
      if (action === "add_task") {
        const title = readStringParam(params, "title", { required: true });
        const description = readStringParam(params, "description", { required: true });
        const fingerprint = readStringParam(params, "fingerprint", { required: true });
        const goalId = readStringParam(params, "goal_id");
        const priority = readStringParam(params, "priority") as
          | "urgent"
          | "high"
          | "normal"
          | "low"
          | undefined;
        const { task, created } = await enqueueTaskIdempotent(agentId, {
          fingerprint,
          goalId,
          title,
          description,
          priority,
        });
        return jsonResult({ task, created });
      }

      if (action === "list_tasks") {
        const filterStatus = readStringParam(params, "filter_status") as
          | "pending"
          | "running"
          | "paused"
          | "completed"
          | "failed"
          | undefined;
        const tasks = await listTasks(agentId, filterStatus ? { status: filterStatus } : undefined);
        return jsonResult({ tasks });
      }

      if (action === "complete_task") {
        const taskId = readStringParam(params, "task_id", { required: true });
        const result = readStringParam(params, "result");
        await completeTask(agentId, taskId, result);
        return jsonResult({ completed: taskId });
      }

      if (action === "pause_task") {
        const taskId = readStringParam(params, "task_id", { required: true });
        await pauseTask(agentId, taskId);
        return jsonResult({ paused: taskId });
      }

      return jsonResult({ error: `unknown action: ${action}` });
    },
  };
}

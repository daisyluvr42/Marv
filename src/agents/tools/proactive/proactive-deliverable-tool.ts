import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../../core/config/config.js";
import {
  getPendingAnnouncements,
  listDeliverables,
  markAnnounced,
  registerDeliverable,
} from "../../../proactive/deliverables.js";
import { resolveSessionAgentId } from "../../agent-scope.js";
import { stringEnum } from "../../schema/typebox.js";
import type { AnyAgentTool } from "../common.js";
import { jsonResult, readStringParam } from "../common.js";

const ACTIONS = ["register", "list", "pending", "mark_announced"] as const;

const DeliverableSchema = Type.Object({
  action: stringEnum(ACTIONS),
  deliverable_id: Type.Optional(Type.String()),
  task_id: Type.Optional(Type.String()),
  goal_id: Type.Optional(Type.String()),
  title: Type.Optional(Type.String()),
  kind: Type.Optional(Type.String()),
  file_path: Type.Optional(Type.String()),
  content_hash: Type.Optional(Type.String()),
  filter_status: Type.Optional(Type.String()),
});

export function createDeliverableTool(options: {
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
    label: "Deliverables",
    name: "deliverable",
    description:
      "Register and track proactive work products. Deliverables follow a lifecycle: stored → announced. Register on task completion; a separate step handles user notification.",
    parameters: DeliverableSchema,
    execute: async (_toolCallId, params) => {
      const action = readStringParam(params, "action", { required: true });

      if (action === "register") {
        const title = readStringParam(params, "title", { required: true });
        const kind = (readStringParam(params, "kind") ?? "other") as
          | "report"
          | "article"
          | "tool"
          | "plan"
          | "analysis"
          | "other";
        const taskId = readStringParam(params, "task_id");
        const goalId = readStringParam(params, "goal_id");
        const filePath = readStringParam(params, "file_path");
        const contentHash = readStringParam(params, "content_hash");
        const { deliverable, created } = await registerDeliverable(agentId, {
          taskId,
          goalId,
          title,
          kind,
          filePath,
          contentHash,
        });
        return jsonResult({ deliverable, created });
      }

      if (action === "list") {
        const filterStatus = readStringParam(params, "filter_status") as
          | "stored"
          | "announced"
          | "failed"
          | undefined;
        const deliverables = await listDeliverables(
          agentId,
          filterStatus ? { status: filterStatus } : undefined,
        );
        return jsonResult({ deliverables });
      }

      if (action === "pending") {
        const pending = await getPendingAnnouncements(agentId);
        return jsonResult({ pending, count: pending.length });
      }

      if (action === "mark_announced") {
        const deliverableId = readStringParam(params, "deliverable_id", { required: true });
        await markAnnounced(agentId, deliverableId);
        return jsonResult({ announced: deliverableId });
      }

      return jsonResult({ error: `unknown action: ${action}` });
    },
  };
}

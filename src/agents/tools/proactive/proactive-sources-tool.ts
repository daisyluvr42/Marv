import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../../core/config/config.js";
import {
  addSource,
  getRecentEvents,
  getSourcesDueForPolling,
  listSources,
  recordPollResult,
  removeSource,
  updateSource,
} from "../../../proactive/sources.js";
import { resolveSessionAgentId } from "../../agent-scope.js";
import { stringEnum } from "../../schema/typebox.js";
import type { AnyAgentTool } from "../common.js";
import { jsonResult, readStringParam } from "../common.js";

const ACTIONS = [
  "add_source",
  "list_sources",
  "update_source",
  "remove_source",
  "recent_events",
  "due_for_polling",
  "record_poll_result",
] as const;

const InfoSourcesSchema = Type.Object({
  action: stringEnum(ACTIONS),
  source_id: Type.Optional(Type.String()),
  kind: Type.Optional(Type.String()),
  label: Type.Optional(Type.String()),
  url: Type.Optional(Type.String()),
  poll_interval_minutes: Type.Optional(Type.Number()),
  enabled: Type.Optional(Type.Boolean()),
  /** Event ID for record_poll_result. */
  event_id: Type.Optional(Type.String()),
  /** Event summary for record_poll_result. */
  event_summary: Type.Optional(Type.String()),
  /** Event detail for record_poll_result. */
  event_detail: Type.Optional(Type.String()),
  /** ISO timestamp of the external event. */
  event_time: Type.Optional(Type.String()),
});

export function createInfoSourcesTool(options: {
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
    label: "Info Sources",
    name: "info_sources",
    description:
      "Manage information sources for proactive monitoring. Use due_for_polling to find sources ready to poll, fetch their content with web_fetch, then record_poll_result to store events.",
    parameters: InfoSourcesSchema,
    execute: async (_toolCallId, params) => {
      const action = readStringParam(params, "action", { required: true });

      if (action === "add_source") {
        const kind = readStringParam(params, "kind", { required: true }) as
          | "rss"
          | "web"
          | "email"
          | "api";
        const label = readStringParam(params, "label", { required: true });
        const url = readStringParam(params, "url");
        const pollMinutes =
          typeof params.poll_interval_minutes === "number"
            ? params.poll_interval_minutes
            : undefined;
        const source = await addSource(agentId, {
          kind,
          label,
          url,
          pollIntervalMs: pollMinutes ? pollMinutes * 60_000 : undefined,
        });
        return jsonResult({ created: source });
      }

      if (action === "list_sources") {
        const sources = await listSources(agentId);
        return jsonResult({ sources });
      }

      if (action === "update_source") {
        const sourceId = readStringParam(params, "source_id", { required: true });
        const patch: Record<string, unknown> = {};
        const label = readStringParam(params, "label");
        const url = readStringParam(params, "url");
        if (label) {
          patch.label = label;
        }
        if (url) {
          patch.url = url;
        }
        if (typeof params.enabled === "boolean") {
          patch.enabled = params.enabled;
        }
        if (typeof params.poll_interval_minutes === "number") {
          patch.pollIntervalMs = params.poll_interval_minutes * 60_000;
        }
        const updated = await updateSource(agentId, sourceId, patch);
        if (!updated) {
          return jsonResult({ error: "source not found" });
        }
        return jsonResult({ updated });
      }

      if (action === "remove_source") {
        const sourceId = readStringParam(params, "source_id", { required: true });
        const removed = await removeSource(agentId, sourceId);
        return jsonResult({ removed });
      }

      if (action === "recent_events") {
        const sourceId = readStringParam(params, "source_id");
        const events = await getRecentEvents(agentId, sourceId ? { sourceId } : undefined);
        return jsonResult({ events, count: events.length });
      }

      if (action === "due_for_polling") {
        const due = await getSourcesDueForPolling(agentId);
        return jsonResult({ sources: due, count: due.length });
      }

      if (action === "record_poll_result") {
        const sourceId = readStringParam(params, "source_id", { required: true });
        const eventId = readStringParam(params, "event_id", { required: true });
        const summary = readStringParam(params, "event_summary", { required: true });
        const detail = readStringParam(params, "event_detail");
        const eventTime = readStringParam(params, "event_time");
        const { newEvents } = await recordPollResult(agentId, sourceId, [
          { id: eventId, sourceId, summary, detail, eventTime },
        ]);
        return jsonResult({ recorded: newEvents.length, deduplicated: newEvents.length === 0 });
      }

      return jsonResult({ error: `unknown action: ${action}` });
    },
  };
}

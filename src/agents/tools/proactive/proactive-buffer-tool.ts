import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../../core/config/config.js";
import {
  appendToDigestBuffer,
  flushDigestBuffer,
  readDigestBuffer,
} from "../../../proactive/digest-buffer.js";
import { resolveSessionAgentId } from "../../agent-scope.js";
import { optionalStringEnum, stringEnum } from "../../schema/typebox.js";
import type { AnyAgentTool } from "../common.js";
import { jsonResult, readStringParam } from "../common.js";

const PROACTIVE_BUFFER_ACTIONS = ["add", "list", "flush"] as const;
const PROACTIVE_BUFFER_URGENCY = ["normal", "urgent"] as const;

const ProactiveBufferSchema = Type.Object({
  action: stringEnum(PROACTIVE_BUFFER_ACTIONS),
  summary: Type.Optional(Type.String()),
  detail: Type.Optional(Type.String()),
  source: Type.Optional(Type.String()),
  urgency: optionalStringEnum(PROACTIVE_BUFFER_URGENCY),
});

export function createProactiveBufferTool(options: {
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
    label: "Proactive Buffer",
    name: "proactive_buffer",
    description:
      "Store, inspect, and flush proactive digest entries during managed proactive runs.",
    parameters: ProactiveBufferSchema,
    execute: async (_toolCallId, params) => {
      const action = readStringParam(params, "action", { required: true });
      if (action === "list") {
        return jsonResult(await readDigestBuffer(agentId));
      }
      if (action === "flush") {
        const entries = await flushDigestBuffer(agentId);
        return jsonResult({
          entries,
          count: entries.length,
        });
      }

      const summary = readStringParam(params, "summary", { required: true });
      const detail = readStringParam(params, "detail");
      const source = readStringParam(params, "source") ?? "unknown";
      const urgency = readStringParam(params, "urgency") === "urgent" ? "urgent" : "normal";
      const entry = await appendToDigestBuffer(agentId, {
        source,
        summary,
        detail: detail ?? undefined,
        urgency,
      });
      return jsonResult({
        ok: true,
        entry,
      });
    },
  };
}

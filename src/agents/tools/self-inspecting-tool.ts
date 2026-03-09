import { Type } from "@sinclair/typebox";
import { loadConfig, type MarvConfig } from "../../core/config/config.js";
import { loadSessionStore, resolveStorePath } from "../../core/config/sessions.js";
import { resolveSessionModelRef } from "../../core/gateway/session-utils.js";
import { resolveAgentIdFromSessionKey } from "../../routing/session-key.js";
import { resolveAgentDir } from "../agent-scope.js";
import {
  cleanupContextPollution,
  inspectContextPollution,
  summarizeContextPollution,
} from "../context-pollution-cleanup.js";
import { resolveRuntimeModelPlan } from "../model/model-pool.js";
import { resolveDefaultModelForAgent } from "../model/model-selection.js";
import type { AnyAgentTool } from "./common.js";
import { ToolInputError } from "./common.js";
import { createSessionStatusTool } from "./session-status-tool.js";

const SelfInspectingToolSchema = Type.Object(
  {
    query: Type.Optional(Type.String()),
    cleanupContextPollution: Type.Optional(Type.Boolean()),
  },
  { additionalProperties: false },
);

type SelfInspectingQuery = "summary" | "runtime" | "models" | "context" | "tools" | "all";

function normalizeQuery(raw: unknown): SelfInspectingQuery {
  const value = typeof raw === "string" ? raw.trim().toLowerCase() : "";
  switch (value) {
    case "runtime":
    case "models":
    case "context":
    case "tools":
    case "all":
      return value;
    default:
      return "summary";
  }
}

function formatModelsSection(params: {
  currentModel: { provider: string; model: string };
  defaultModel: { provider: string; model: string };
  runtimePlan: ReturnType<typeof resolveRuntimeModelPlan>;
}): string {
  const lines = [
    `Current model: ${params.currentModel.provider}/${params.currentModel.model}`,
    `Default model: ${params.defaultModel.provider}/${params.defaultModel.model}`,
    `Model pool: ${params.runtimePlan.poolName}`,
  ];
  if (params.runtimePlan.candidates.length > 0) {
    lines.push(
      `Runnable candidates: ${params.runtimePlan.candidates.map((entry) => entry.ref).join(", ")}`,
    );
  }
  const unavailable = params.runtimePlan.configured
    .filter((entry) => !entry.available)
    .map((entry) =>
      entry.availabilityReason ? `${entry.ref} (${entry.availabilityReason})` : entry.ref,
    );
  if (unavailable.length > 0) {
    lines.push(`Unavailable configured models: ${unavailable.join(", ")}`);
  }
  return lines.join("\n");
}

function formatToolsSection(params: {
  availableToolNames: string[];
  directUserInstruction?: boolean;
}): string {
  const lines = [
    `Available tools (${params.availableToolNames.length}): ${params.availableToolNames.join(", ")}`,
  ];
  if (params.directUserInstruction === false) {
    lines.push(
      "Some self-modifying actions are currently blocked because the instruction is not direct user input.",
    );
  }
  return lines.join("\n");
}

function extractTextBlocks(content: unknown): string {
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .filter(
      (block): block is { type: "text"; text: string } =>
        typeof block === "object" &&
        block !== null &&
        "type" in block &&
        "text" in block &&
        block.type === "text" &&
        typeof block.text === "string",
    )
    .map((block) => block.text.trim())
    .filter(Boolean)
    .join("\n\n");
}

export function createSelfInspectingTool(opts?: {
  agentSessionKey?: string;
  config?: MarvConfig;
  availableToolNames?: string[];
  directUserInstruction?: boolean;
}): AnyAgentTool {
  return {
    label: "Self Inspecting",
    name: "self_inspecting",
    description:
      "Inspect your own runtime, models, context, and tool state. Use when the user asks about your current status, available models, tools, limits, or why you are behaving a certain way. Optional: cleanupContextPollution=true removes recognized polluted assistant history for the current session.",
    parameters: SelfInspectingToolSchema,
    execute: async (_toolCallId, args) => {
      const cfg = opts?.config ?? loadConfig();
      const sessionKey = opts?.agentSessionKey?.trim();
      if (!sessionKey) {
        throw new ToolInputError("sessionKey required");
      }

      const query = normalizeQuery((args as Record<string, unknown>).query);
      const cleanupRequested = Boolean((args as Record<string, unknown>).cleanupContextPollution);
      if (cleanupRequested && opts?.directUserInstruction === false) {
        return {
          content: [
            {
              type: "text",
              text: "I can't clean my own context from an indirect or forwarded instruction.",
            },
          ],
          details: { ok: false, denied: true, query, cleanupRequested },
        };
      }

      const agentId = resolveAgentIdFromSessionKey(sessionKey);
      const storePath = resolveStorePath(cfg.session?.store, { agentId });
      const store = loadSessionStore(storePath, { skipCache: true });
      const sessionEntry = store[sessionKey];
      const defaultModel = resolveDefaultModelForAgent({ cfg, agentId });
      const currentModel = resolveSessionModelRef(cfg, sessionEntry, agentId);
      const runtimePlan = resolveRuntimeModelPlan({
        cfg,
        agentId,
        agentDir: resolveAgentDir(cfg, agentId),
      });
      const sessionStatusTool = createSessionStatusTool({
        agentSessionKey: sessionKey,
        config: cfg,
      });
      const sessionStatusResult = await sessionStatusTool.execute("self-inspecting-runtime", {
        sessionKey,
      });
      const sessionStatusText =
        (
          sessionStatusResult.details as {
            statusText?: string;
          }
        )?.statusText ??
        extractTextBlocks(sessionStatusResult.content) ??
        "Session status unavailable.";

      const inspection = inspectContextPollution({
        cfg,
        sessionKey,
      });
      const cleanupResult = cleanupRequested
        ? await cleanupContextPollution({
            cfg,
            sessionKey,
          })
        : null;

      const modelsSection = formatModelsSection({
        currentModel,
        defaultModel,
        runtimePlan,
      });
      const contextSection = [
        summarizeContextPollution(cleanupResult ?? inspection),
        cleanupResult
          ? `Cleanup removed transcript ${cleanupResult.cleaned.transcriptRemoved}, task context ${cleanupResult.cleaned.taskContextRemoved}.`
          : undefined,
      ]
        .filter(Boolean)
        .join("\n");
      const toolsSection = formatToolsSection({
        availableToolNames: [...new Set(opts?.availableToolNames ?? [])].toSorted(),
        directUserInstruction: opts?.directUserInstruction,
      });

      let text: string;
      switch (query) {
        case "runtime":
          text = sessionStatusText || "Session status unavailable.";
          break;
        case "models":
          text = modelsSection;
          break;
        case "context":
          text = contextSection;
          break;
        case "tools":
          text = toolsSection;
          break;
        case "all":
          text = [
            "Runtime",
            sessionStatusText || "Session status unavailable.",
            "",
            "Models",
            modelsSection,
            "",
            "Context",
            contextSection,
            "",
            "Tools",
            toolsSection,
          ].join("\n");
          break;
        case "summary":
        default:
          text = [
            `Current model: ${currentModel.provider}/${currentModel.model}`,
            `Default model: ${defaultModel.provider}/${defaultModel.model}`,
            `Model pool: ${runtimePlan.poolName} (${runtimePlan.candidates.length} runnable)`,
            summarizeContextPollution(cleanupResult ?? inspection),
          ].join("\n");
          break;
      }

      return {
        content: [{ type: "text", text }],
        details: {
          ok: true,
          query,
          cleanupRequested,
          cleaned: cleanupResult?.cleaned ?? null,
          runtime: {
            sessionKey,
            agentId,
            currentModel: `${currentModel.provider}/${currentModel.model}`,
            defaultModel: `${defaultModel.provider}/${defaultModel.model}`,
            sessionStatusText,
          },
          models: {
            poolName: runtimePlan.poolName,
            candidates: runtimePlan.candidates.map((entry) => entry.ref),
            configured: runtimePlan.configured.map((entry) => ({
              ref: entry.ref,
              available: entry.available,
              availabilityReason: entry.availabilityReason,
            })),
          },
          context: {
            preferences: (cleanupResult ?? inspection).preferences,
            transcriptViolations: (cleanupResult ?? inspection).transcript.violations.length,
            transcriptRemovable:
              (cleanupResult ?? inspection).transcript.removableIds.length +
              (cleanupResult ?? inspection).transcript.sanitizedIds.length,
            taskContextViolations: (cleanupResult ?? inspection).taskContext.violations.length,
            taskContextRemovable: (cleanupResult ?? inspection).taskContext.removableIds.length,
          },
          tools: {
            available: [...new Set(opts?.availableToolNames ?? [])].toSorted(),
          },
        },
      };
    },
  };
}

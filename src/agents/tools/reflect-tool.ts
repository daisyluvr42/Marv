import type { MarvConfig } from "../../core/config/config.js";
import { distillExperience } from "../../memory/experience/experience-distiller.js";
import {
  writeExperienceFile,
  CONTEXT_BUDGET_CHARS,
  measureExperienceContent,
} from "../../memory/experience/experience-files.js";
import { loadSoulFile } from "../soul.js";
import { jsonResult } from "./common.js";
import type { AnyAgentTool } from "./common.js";

/**
 * JSON schema for the reflect tool input.
 */
const ReflectSchema = {
  type: "object",
  properties: {
    target: {
      type: "string",
      enum: ["experience", "context"],
      description:
        "experience: trigger experience distillation from behavioral lessons; context: update current working context",
    },
    content: {
      type: "string",
      description: "The content to record",
    },
  },
  required: ["target", "content"],
} as const;

/**
 * Create the reflect tool for an agent.
 * - target=experience: triggers LLM distillation into MARV_EXPERIENCE.md
 * - target=context: directly writes to MARV_CONTEXT.md
 */
export function createReflectTool(options: {
  config?: MarvConfig;
  agentSessionKey?: string;
}): AnyAgentTool | null {
  const agentId = extractAgentId(options.agentSessionKey);
  if (!agentId) {
    return null;
  }

  return {
    label: "Reflect",
    name: "reflect",
    description:
      "Record behavioral experience and lessons (target=experience, triggers LLM distillation) or update current working context (target=context).",
    parameters: ReflectSchema,
    execute: async (_toolCallId, params) => {
      const target = readParam(params, "target");
      const content = readParam(params, "content");

      if (!target || !content) {
        return jsonResult({ ok: false, error: "target and content are required" });
      }

      if (target === "experience") {
        const soulContent = await loadSoulFile(agentId);
        const result = await distillExperience(
          agentId,
          {
            source: "reflect",
            content,
            timestamp: Date.now(),
          },
          {
            cfg: options.config,
            soulContent,
            force: true, // reflect always bypasses debounce
          },
        );
        return jsonResult({
          ok: true,
          updated: result.updated,
          message: result.updated
            ? "Experience has been updated"
            : "No update needed for current experience document",
          changes: result.changes,
        });
      }

      if (target === "context") {
        const budget =
          options.config?.memory?.experience?.contextBudgetChars ?? CONTEXT_BUDGET_CHARS;
        if (measureExperienceContent(content) > budget) {
          return jsonResult({
            ok: false,
            error: `Context content exceeds budget of ${budget} characters`,
          });
        }
        await writeExperienceFile(agentId, "MARV_CONTEXT.md", content);
        return jsonResult({
          ok: true,
          message: "Working context has been updated",
        });
      }

      return jsonResult({ ok: false, error: `Unknown target: ${target}` });
    },
  };
}

// --- Internal helpers ---

function extractAgentId(sessionKey?: string): string | null {
  if (!sessionKey) {
    return null;
  }
  // Session keys are formatted as "agent:main:..." or similar
  const parts = sessionKey.split(":");
  if (parts.length >= 2 && parts[0] === "agent") {
    return parts[1] ?? null;
  }
  // Fallback: use the entire sessionKey as agentId
  return sessionKey;
}

function readParam(params: Record<string, unknown>, key: string): string | null {
  const value = params[key];
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  return null;
}

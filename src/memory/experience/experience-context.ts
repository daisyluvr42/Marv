import type { MarvConfig } from "../../core/config/config.js";
import {
  writeExperienceFile,
  readExperienceFile,
  measureExperienceContent,
  CONTEXT_BUDGET_CHARS,
} from "./experience-files.js";

/**
 * Distill session context into MARV_CONTEXT.md.
 * Uses a local/cheap model to summarize working context and progress.
 *
 * Called at session end, overflow, or when the agent explicitly reflects on context.
 */
export async function distillSessionContext(params: {
  agentId: string;
  sessionSummary: string;
  cfg?: MarvConfig;
}): Promise<void> {
  const budget = params.cfg?.memory?.experience?.contextBudgetChars ?? CONTEXT_BUDGET_CHARS;

  // Use dynamic model selection: picks a cheap/local model for context summarization
  try {
    const { experienceInfer } = await import("./experience-inference.js");

    const result = await experienceInfer({
      cfg: params.cfg ?? {},
      role: "context",
      system:
        "Summarize the following session into key working context points and progress. " +
        `Keep it concise, under ${budget} characters. ` +
        "Focus on: current task state, important decisions made, pending items, and any blockers.",
      prompt: params.sessionSummary,
      agentId: params.agentId,
    });

    if (result.ok) {
      let content = result.text.trim();
      if (measureExperienceContent(content) > budget) {
        content = content.slice(0, budget);
      }
      await writeExperienceFile(params.agentId, "MARV_CONTEXT.md", content);
      return;
    }
  } catch {
    // LLM not available, use heuristic
  }

  // Heuristic fallback: truncate to budget
  let content = params.sessionSummary.trim();
  if (measureExperienceContent(content) > budget) {
    content = content.slice(0, budget);
  }
  await writeExperienceFile(params.agentId, "MARV_CONTEXT.md", content);
}

/**
 * Check if CONTEXT.md is stale (from a previous session).
 * Returns true if the file is older than the given threshold.
 */
export async function isContextStale(
  agentId: string,
  maxAgeMsDefault = 4 * 60 * 60 * 1000, // 4 hours
): Promise<boolean> {
  const content = await readExperienceFile(agentId, "MARV_CONTEXT.md");
  if (!content.trim()) {
    return true; // Empty is considered stale
  }

  // Check file mtime
  try {
    const { resolveExperienceDir } = await import("./experience-files.js");
    const { stat } = await import("node:fs/promises");
    const path = await import("node:path");
    const filePath = path.join(resolveExperienceDir(agentId), "MARV_CONTEXT.md");
    const stats = await stat(filePath);
    return Date.now() - stats.mtimeMs > maxAgeMsDefault;
  } catch {
    return true; // File doesn't exist or can't be read
  }
}

/**
 * Clear stale context at session start.
 */
export async function clearStaleContext(agentId: string): Promise<void> {
  const stale = await isContextStale(agentId);
  if (stale) {
    await writeExperienceFile(agentId, "MARV_CONTEXT.md", "");
  }
}

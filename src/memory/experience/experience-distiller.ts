import type { MarvConfig } from "../../core/config/config.js";
import type { ExperienceConfig } from "../../core/config/types.memory.js";
import {
  readExperienceFile,
  writeExperienceFile,
  appendExperienceLog,
  measureExperienceContent,
  EXPERIENCE_BUDGET_CHARS,
} from "./experience-files.js";

/**
 * Input to the distillation process.
 */
export type DistillationInput = {
  source: "overflow" | "task_completion" | "reflect" | "goal_strategy";
  content: string;
  timestamp: number;
};

/**
 * Result of a distillation attempt.
 */
export type DistillationResult = {
  updated: boolean;
  changes?: string;
  skillCandidates?: string[];
};

/**
 * Debounce state per agent.
 */
const lastDistillTimestamps = new Map<string, number>();

const DEFAULT_DEBOUNCE_MS = 4 * 60 * 60 * 1000; // 4 hours

const DISTILL_SYSTEM_PROMPT = `You are an experience distillation engine. Your task is to maintain an experience document.

Input:
1. Current experience document content (each entry may have a:activation_count p:positive_count markers)
2. Soul principles (must not be violated)
3. New data (conversation fragments/task results/reflection records)

Rules:
- Only record behavioral experience and lessons, not factual knowledge
- If no experience worth recording in the new data, output NO_UPDATE
- Do not add duplicates of existing experience
- If new data contradicts existing experience, update the old entry and explain why
- If new data contains reusable skill patterns, mark as SKILL_CANDIDATE
- Must not violate Soul principles
- Total content must not exceed {budget} characters
- [Attribution-aware] Pay attention to each experience's a/p markers:
  - a:0 entries may be poorly worded and never activated, consider rewriting for applicability
  - Low p/a ratio (e.g. a:10 p:2) suggests the experience may be wrong, consider correction with new data
  - High p/a ratio (e.g. a:8 p:7) indicates highly effective experience, prioritize retention
  - New entries should be initialized with a:0 p:0

Output format:
ACTION: UPDATE | NO_UPDATE
CHANGES: [added: "...", updated: "old → new", removed: "..."]
SKILL_CANDIDATES: [if any]
---
[Updated complete experience document, preserving a:N p:N markers on each entry]`;

/**
 * Distill experience from new data into MARV_EXPERIENCE.md.
 * Includes debounce logic to prevent excessive LLM calls.
 *
 * In the current implementation, uses a heuristic approach as a fallback
 * when no LLM is available. When LLM is available, delegates to model.
 */
export async function distillExperience(
  agentId: string,
  newData: DistillationInput,
  options?: {
    cfg?: MarvConfig;
    soulContent?: string;
    /** Force distillation, bypassing debounce. */
    force?: boolean;
  },
): Promise<DistillationResult> {
  const experienceConfig = resolveExperienceConfig(options?.cfg);

  // Check debounce (skip for forced calls and reflect source)
  if (!options?.force && newData.source !== "reflect") {
    const lastTimestamp = lastDistillTimestamps.get(agentId) ?? 0;
    const debounceMs = experienceConfig.distillDebounceMs ?? DEFAULT_DEBOUNCE_MS;
    if (Date.now() - lastTimestamp < debounceMs) {
      // Log the input but skip distillation
      await appendDistillLog(agentId, newData, "debounced");
      return { updated: false };
    }
  }

  const currentExperience = await readExperienceFile(agentId, "MARV_EXPERIENCE.md");
  const budget = experienceConfig.experienceBudgetChars ?? EXPERIENCE_BUDGET_CHARS;

  // Try LLM-based distillation
  const result = await distillWithLlm({
    agentId,
    currentExperience,
    soulContent: options?.soulContent ?? "",
    newData,
    budget,
    cfg: options?.cfg,
  });

  if (!result.updated) {
    await appendDistillLog(agentId, newData, "unchanged");
    return { updated: false };
  }

  // Write updated experience
  await writeExperienceFile(agentId, "MARV_EXPERIENCE.md", result.newExperience);

  // Append to log
  await appendDistillLog(agentId, newData, result.action, result.changes);

  // Update debounce timestamp
  lastDistillTimestamps.set(agentId, Date.now());

  return {
    updated: true,
    changes: result.changes,
    skillCandidates: result.skillCandidates,
  };
}

/**
 * Enqueue a distillation request (fire-and-forget, non-blocking).
 * Used by overflow and task completion paths to avoid blocking the main flow.
 */
export function enqueueDistillation(
  agentId: string,
  input: DistillationInput,
  options?: { cfg?: MarvConfig; soulContent?: string },
): void {
  // Fire-and-forget: Promise rejection is caught and logged
  distillExperience(agentId, input, options).catch((err) => {
    // Log but don't crash
    console.error(`[experience-distiller] distillation failed for ${agentId}:`, err);
  });
}

// --- Internal helpers ---

type LlmDistillResult = {
  updated: boolean;
  action: string;
  newExperience: string;
  changes?: string;
  skillCandidates?: string[];
};

async function distillWithLlm(params: {
  agentId: string;
  currentExperience: string;
  soulContent: string;
  newData: DistillationInput;
  budget: number;
  cfg?: MarvConfig;
}): Promise<LlmDistillResult> {
  // Try to use local LLM for distillation
  try {
    const { inferLocal } = await import("../storage/local-llm-client.js");

    const modelConfig = params.cfg?.memory?.soul?.deepConsolidation?.model;
    const systemPrompt = DISTILL_SYSTEM_PROMPT.replace("{budget}", String(params.budget));
    const userPrompt = buildDistillUserPrompt(params);

    const result = await inferLocal({
      cfg: params.cfg ?? {},
      model: modelConfig,
      system: systemPrompt,
      prompt: userPrompt,
    });

    if (!result.ok) {
      // LLM failed, fall back to heuristic
      return distillWithHeuristic(params);
    }

    return parseDistillOutput(result.text, params.budget);
  } catch {
    // No LLM available, use heuristic fallback
    return distillWithHeuristic(params);
  }
}

function buildDistillUserPrompt(params: {
  currentExperience: string;
  soulContent: string;
  newData: DistillationInput;
}): string {
  const parts = [
    "## Current Experience Document",
    params.currentExperience || "(empty)",
    "",
    "## Soul Principles",
    params.soulContent || "(none configured)",
    "",
    `## New Data (source: ${params.newData.source})`,
    params.newData.content,
  ];
  return parts.join("\n");
}

function parseDistillOutput(text: string, budget: number): LlmDistillResult {
  const lines = text.split("\n");
  const actionLine = lines.find((l) => l.startsWith("ACTION:"));
  const action = actionLine?.replace("ACTION:", "").trim() ?? "NO_UPDATE";

  if (action === "NO_UPDATE") {
    return { updated: false, action: "no_update", newExperience: "" };
  }

  const changesLine = lines.find((l) => l.startsWith("CHANGES:"));
  const changes = changesLine?.replace("CHANGES:", "").trim();

  const skillLine = lines.find((l) => l.startsWith("SKILL_CANDIDATES:"));
  const skillCandidates = skillLine
    ? skillLine
        .replace("SKILL_CANDIDATES:", "")
        .trim()
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
    : undefined;

  // Extract the experience document after the --- separator
  const separatorIdx = text.indexOf("---");
  let newExperience = separatorIdx >= 0 ? text.slice(separatorIdx + 3).trim() : "";

  // Budget check
  if (measureExperienceContent(newExperience) > budget) {
    // Truncate to budget (crude; LLM should compress, but this is a safety net)
    newExperience = newExperience.slice(0, budget);
  }

  return {
    updated: true,
    action: "update",
    newExperience,
    changes,
    skillCandidates,
  };
}

/**
 * Heuristic fallback when no LLM is available.
 * Simply appends the new data as a new experience entry.
 */
function distillWithHeuristic(params: {
  currentExperience: string;
  newData: DistillationInput;
  budget: number;
}): LlmDistillResult {
  const content = params.newData.content.trim();
  if (!content) {
    return { updated: false, action: "no_update", newExperience: "" };
  }

  // Extract the first meaningful sentence as an experience entry
  const firstSentence = content.split(/[.!?\n]/)[0]?.trim();
  if (!firstSentence || firstSentence.length < 10) {
    return { updated: false, action: "no_update", newExperience: "" };
  }

  const dateTag = new Date().toISOString().slice(0, 7); // YYYY-MM
  const newEntry = `[LEARNED] ${firstSentence} @${dateTag} | a:0 p:0`;

  let updated = params.currentExperience.trim();
  if (updated) {
    updated += `\n${newEntry}`;
  } else {
    updated = `## Experience\n\n${newEntry}`;
  }

  // Budget check
  if (measureExperienceContent(updated) > params.budget) {
    return { updated: false, action: "no_update", newExperience: "" };
  }

  return {
    updated: true,
    action: "update",
    newExperience: updated,
    changes: `added: "${firstSentence.slice(0, 50)}..."`,
  };
}

async function appendDistillLog(
  agentId: string,
  input: DistillationInput,
  action: string,
  changes?: string,
): Promise<void> {
  const timestamp = new Date(input.timestamp).toISOString().replace("T", " ").slice(0, 16);
  const logEntry = `[${timestamp}] [${input.source}] ${input.content.slice(0, 100).replace(/\n/g, " ")} | ${action}${changes ? `: ${changes}` : ""}`;
  await appendExperienceLog(agentId, logEntry);
}

function resolveExperienceConfig(cfg?: MarvConfig): Partial<ExperienceConfig> {
  return cfg?.memory?.experience ?? {};
}

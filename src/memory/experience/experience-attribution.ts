import type { MarvConfig } from "../../core/config/config.js";
import { readExperienceFileSync } from "./experience-files.js";

/**
 * Represents a single experience entry parsed from EXPERIENCE.md.
 */
export type ExperienceEntry = {
  /** Content hash (first 8 chars) for identification. */
  id: string;
  /** The full line text. */
  text: string;
  /** Activation count (a:N). */
  activationCount: number;
  /** Positive count (p:N). */
  positiveCount: number;
};

export type TaskOutcome = "completed" | "advancing" | "stalled" | "ambiguous";

export type AttributionResult = {
  activatedEntries: Array<{
    entryId: string;
    confidence: number;
  }>;
  outcome: TaskOutcome;
};

/**
 * Parse EXPERIENCE.md into structured entries.
 */
export function parseExperienceEntries(content: string): ExperienceEntry[] {
  const entries: ExperienceEntry[] = [];
  const pattern =
    /\[(?:LEARNED|AVOID|PREF|SKILL)\]\s+(.+?)(?:\s+@\d{4}-\d{2})?\s*\|\s*a:(\d+)\s+p:(\d+)/g;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(content)) !== null) {
    const text = match[0];
    const a = parseInt(match[2] ?? "0", 10);
    const p = parseInt(match[3] ?? "0", 10);
    // Use a simple hash of the content for identification
    const id = simpleHash(match[1] ?? text);
    entries.push({
      id,
      text,
      activationCount: a,
      positiveCount: p,
    });
  }

  return entries;
}

/**
 * Detect which experience entries were activated during the agent's response.
 *
 * Uses a two-stage approach:
 * 1. Text similarity prefilter (cheap, ~5ms)
 * 2. LLM confirmation (only for candidates, cheap model)
 */
export async function detectActivatedExperiences(params: {
  agentId: string;
  agentResponse: string;
  taskOutcome: TaskOutcome;
  cfg?: MarvConfig;
}): Promise<AttributionResult> {
  const experience = readExperienceFileSync(params.agentId, "MARV_EXPERIENCE.md");
  if (!experience.trim()) {
    return { activatedEntries: [], outcome: params.taskOutcome };
  }

  const entries = parseExperienceEntries(experience);
  if (entries.length === 0) {
    return { activatedEntries: [], outcome: params.taskOutcome };
  }

  // Stage 1: Text similarity prefilter
  const candidates = prefilterByTextSimilarity(entries, params.agentResponse, 0.15);
  if (candidates.length === 0) {
    return { activatedEntries: [], outcome: params.taskOutcome };
  }

  // Stage 2: LLM confirmation (if available, uses dynamic model selection)
  const confirmed = await confirmActivation(
    candidates,
    params.agentResponse,
    params.cfg,
    params.agentId,
  );

  return {
    activatedEntries: confirmed.map((c) => ({
      entryId: c.id,
      confidence: c.confidence,
    })),
    outcome: params.taskOutcome,
  };
}

// --- Internal helpers ---

type ScoredCandidate = ExperienceEntry & { similarity: number };
type ConfirmedCandidate = ExperienceEntry & { confidence: number };

/**
 * Prefilter by simple text overlap (word-level Jaccard similarity).
 * Much cheaper than embedding similarity but sufficient for a prefilter.
 */
function prefilterByTextSimilarity(
  entries: ExperienceEntry[],
  response: string,
  threshold: number,
): ScoredCandidate[] {
  const responseTokens = tokenize(response);
  if (responseTokens.size === 0) {
    return [];
  }

  const candidates: ScoredCandidate[] = [];
  for (const entry of entries) {
    const entryTokens = tokenize(entry.text);
    if (entryTokens.size === 0) {
      continue;
    }

    const intersection = new Set([...entryTokens].filter((t) => responseTokens.has(t)));
    const union = new Set([...entryTokens, ...responseTokens]);
    const jaccard = union.size > 0 ? intersection.size / union.size : 0;

    if (jaccard >= threshold) {
      candidates.push({ ...entry, similarity: jaccard });
    }
  }

  return candidates.toSorted((a, b) => b.similarity - a.similarity).slice(0, 5);
}

/**
 * Confirm activation using LLM (cheap model).
 * Falls back to accepting all candidates if LLM is unavailable.
 */
async function confirmActivation(
  candidates: ScoredCandidate[],
  response: string,
  cfg?: MarvConfig,
  agentId?: string,
): Promise<ConfirmedCandidate[]> {
  try {
    const { experienceInfer } = await import("./experience-inference.js");

    const candidateList = candidates.map((c, i) => `${i + 1}. ${c.text}`).join("\n");

    const result = await experienceInfer({
      cfg: cfg ?? {},
      role: "attribution",
      system:
        "You are an attribution detector. Given an agent response and a list of experience entries, " +
        "determine which experiences influenced the response. Output ONLY the numbers of activated entries, " +
        "comma-separated. If none are activated, output NONE.",
      prompt: `Agent response:\n${response.slice(0, 1000)}\n\nCandidate experiences:\n${candidateList}`,
      agentId,
    });

    if (result.ok) {
      const text = result.text.trim();
      if (text === "NONE" || text === "none") {
        return [];
      }
      const activatedIndices = text
        .split(",")
        .map((s) => parseInt(s.trim(), 10) - 1)
        .filter((i) => !isNaN(i) && i >= 0 && i < candidates.length);

      return activatedIndices.map((i) => ({
        ...candidates[i],
        confidence: 0.8, // LLM-confirmed
      }));
    }
  } catch {
    // LLM unavailable
  }

  // Fallback: accept top candidates with moderate confidence
  return candidates.slice(0, 3).map((c) => ({
    ...c,
    confidence: c.similarity,
  }));
}

function tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]/gu, " ")
      .split(/\s+/)
      .filter((t) => t.length > 2),
  );
}

function simpleHash(text: string): string {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = ((hash << 5) - hash + char) | 0;
  }
  return Math.abs(hash).toString(36).slice(0, 8);
}

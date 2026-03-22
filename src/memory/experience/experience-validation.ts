import type { AttributionResult } from "./experience-attribution.js";
import { readExperienceFileSync, writeExperienceFile } from "./experience-files.js";

/**
 * Record experience outcome by updating a/p counts in EXPERIENCE.md.
 * Called after goal-loop judges task progress.
 *
 * - Increments `a:N` for all activated entries
 * - Increments `p:N` for activated entries when outcome is positive
 */
export async function recordExperienceOutcome(
  agentId: string,
  attribution: AttributionResult,
): Promise<void> {
  if (attribution.activatedEntries.length === 0) {
    return;
  }

  const experience = readExperienceFileSync(agentId, "MARV_EXPERIENCE.md");
  if (!experience.trim()) {
    return;
  }

  let updated = experience;
  const isPositive = attribution.outcome === "completed" || attribution.outcome === "advancing";

  for (const entry of attribution.activatedEntries) {
    updated = incrementActivationCount(updated, entry.entryId);
    if (isPositive) {
      updated = incrementPositiveCount(updated, entry.entryId);
    }
  }

  if (updated !== experience) {
    await writeExperienceFile(agentId, "MARV_EXPERIENCE.md", updated);
  }
}

/**
 * Increment the a:N count for a matched experience entry.
 * Matches entries by their content hash prefix.
 */
function incrementActivationCount(content: string, entryId: string): string {
  // Find entries and increment their a: count
  return content.replace(/\| a:(\d+) p:(\d+)/g, (fullMatch, aStr, pStr) => {
    // Check if this line's content hash matches the entryId
    const lineStart = content.lastIndexOf("\n", content.indexOf(fullMatch)) + 1;
    const lineContent = content.slice(lineStart, content.indexOf(fullMatch));
    const lineHash = simpleHash(extractEntryContent(lineContent));

    if (lineHash === entryId) {
      const newA = parseInt(aStr as string, 10) + 1;
      return `| a:${newA} p:${pStr}`;
    }
    return fullMatch;
  });
}

/**
 * Increment the p:N count for a matched experience entry.
 */
function incrementPositiveCount(content: string, entryId: string): string {
  return content.replace(/\| a:(\d+) p:(\d+)/g, (fullMatch, aStr, pStr) => {
    const lineStart = content.lastIndexOf("\n", content.indexOf(fullMatch)) + 1;
    const lineContent = content.slice(lineStart, content.indexOf(fullMatch));
    const lineHash = simpleHash(extractEntryContent(lineContent));

    if (lineHash === entryId) {
      const newP = parseInt(pStr as string, 10) + 1;
      return `| a:${aStr} p:${newP}`;
    }
    return fullMatch;
  });
}

/**
 * Extract the core content from an experience line (between tag and date).
 */
function extractEntryContent(line: string): string {
  const match = /\[(?:LEARNED|AVOID|PREF|SKILL)\]\s+(.+?)(?:\s+@\d{4}-\d{2})?$/.exec(line.trim());
  return match?.[1]?.trim() ?? line.trim();
}

function simpleHash(text: string): string {
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = ((hash << 5) - hash + char) | 0;
  }
  return Math.abs(hash).toString(36).slice(0, 8);
}

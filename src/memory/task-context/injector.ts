import { textSimilarity } from "../search/mmr.js";
import { listSoulMemoryItems, writeSoulMemory } from "../storage/soul-memory-store.js";
import type { DistilledKnowledge } from "./distiller.js";

export type InjectDistilledKnowledgeResult = {
  insertedFacts: number;
  insertedPreferences: number;
  insertedLessons: number;
  insertedSkillsAsLessons: number;
  skippedAsDuplicate: number;
};

export function injectDistilledKnowledge(params: {
  agentId: string;
  taskScopeId: string;
  distilled: DistilledKnowledge;
  duplicateThreshold?: number;
}): InjectDistilledKnowledgeResult {
  const duplicateThreshold = Math.max(0, Math.min(1, params.duplicateThreshold ?? 0.85));
  const existing = listSoulMemoryItems({
    agentId: params.agentId,
    limit: 2000,
  });
  const existingTexts = existing.map((item) => item.content);

  let insertedFacts = 0;
  let insertedPreferences = 0;
  let insertedLessons = 0;
  let insertedSkillsAsLessons = 0;
  let skippedAsDuplicate = 0;

  const shouldSkipDuplicate = (content: string): boolean => {
    const normalized = content.trim();
    if (!normalized) {
      return true;
    }
    for (const existingContent of existingTexts) {
      const similarity = textSimilarity(normalized, existingContent);
      if (similarity >= duplicateThreshold) {
        return true;
      }
    }
    return false;
  };

  const registerInserted = (content: string) => {
    existingTexts.push(content);
  };

  for (const fact of params.distilled.facts) {
    if (shouldSkipDuplicate(fact.content)) {
      skippedAsDuplicate += 1;
      continue;
    }
    const inserted = writeSoulMemory({
      agentId: params.agentId,
      scopeType: "task",
      scopeId: params.taskScopeId,
      kind: fact.kind || "fact",
      content: fact.content,
      confidence: clampConfidence(fact.confidence, 0.45),
      source: "auto_extraction",
    });
    if (inserted) {
      insertedFacts += 1;
      registerInserted(fact.content);
    }
  }

  for (const preference of params.distilled.preferences) {
    if (shouldSkipDuplicate(preference.content)) {
      skippedAsDuplicate += 1;
      continue;
    }
    const inserted = writeSoulMemory({
      agentId: params.agentId,
      scopeType: "task",
      scopeId: params.taskScopeId,
      kind: "preference",
      content: preference.content,
      confidence: clampConfidence(preference.confidence, 0.62),
      source: "manual_log",
    });
    if (inserted) {
      insertedPreferences += 1;
      registerInserted(preference.content);
    }
  }

  for (const lesson of params.distilled.lessons) {
    if (shouldSkipDuplicate(lesson.content)) {
      skippedAsDuplicate += 1;
      continue;
    }
    const inserted = writeSoulMemory({
      agentId: params.agentId,
      scopeType: "task",
      scopeId: params.taskScopeId,
      kind: "lesson",
      content: lesson.content,
      confidence: 0.55,
      source: "auto_extraction",
    });
    if (inserted) {
      insertedLessons += 1;
      registerInserted(lesson.content);
    }
  }

  for (const skill of params.distilled.skills) {
    const skillText =
      `${skill.name}: ${skill.description} Steps: ${skill.steps.join(" -> ")}`.trim();
    if (!skillText || shouldSkipDuplicate(skillText)) {
      skippedAsDuplicate += 1;
      continue;
    }
    const inserted = writeSoulMemory({
      agentId: params.agentId,
      scopeType: "task",
      scopeId: params.taskScopeId,
      kind: "lesson",
      content: skillText,
      confidence: 0.52,
      source: "auto_extraction",
    });
    if (inserted) {
      insertedSkillsAsLessons += 1;
      registerInserted(skillText);
    }
  }

  return {
    insertedFacts,
    insertedPreferences,
    insertedLessons,
    insertedSkillsAsLessons,
    skippedAsDuplicate,
  };
}

function clampConfidence(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(0.05, Math.min(0.99, value));
}

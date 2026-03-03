import fs from "node:fs/promises";
import type { TaskArchive } from "./archiver.js";

export type DistilledKnowledge = {
  facts: Array<{ content: string; kind: string; confidence: number }>;
  skills: Array<{ name: string; description: string; steps: string[] }>;
  preferences: Array<{ content: string; confidence: number }>;
  lessons: Array<{ content: string; context: string }>;
};

type DistillInputChunk = {
  sequence: number;
  role: string;
  content: string;
};

export async function distillTaskContext(params: {
  agentId: string;
  taskId: string;
  archive: TaskArchive;
  distillWithLlm?: (input: {
    chunks: DistillInputChunk[];
    taskId: string;
  }) => Promise<DistilledKnowledge>;
}): Promise<DistilledKnowledge> {
  const chunks = await readArchiveChunks(params.archive.contextJsonlPath);
  if (params.distillWithLlm) {
    const distilled = await params.distillWithLlm({
      chunks,
      taskId: params.taskId,
    });
    return dedupeDistilledKnowledge(distilled);
  }

  const facts = extractFacts(chunks);
  const preferences = extractPreferences(chunks);
  const lessons = extractLessons(chunks);
  const skills = extractSkills(chunks, params.taskId);

  return dedupeDistilledKnowledge({
    facts,
    preferences,
    lessons,
    skills,
  });
}

async function readArchiveChunks(contextJsonlPath: string): Promise<DistillInputChunk[]> {
  const raw = await fs.readFile(contextJsonlPath, "utf-8");
  if (!raw.trim()) {
    return [];
  }
  const lines = raw.split(/\r?\n/).filter(Boolean);
  const chunks: DistillInputChunk[] = [];
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line) as {
        sequence?: unknown;
        role?: unknown;
        content?: unknown;
      };
      if (typeof parsed.content !== "string" || !parsed.content.trim()) {
        continue;
      }
      chunks.push({
        sequence:
          typeof parsed.sequence === "number" && Number.isFinite(parsed.sequence)
            ? Math.max(1, Math.floor(parsed.sequence))
            : chunks.length + 1,
        role: typeof parsed.role === "string" ? parsed.role : "assistant",
        content: parsed.content.trim(),
      });
    } catch {
      continue;
    }
  }
  return chunks;
}

function extractFacts(chunks: DistillInputChunk[]): DistilledKnowledge["facts"] {
  const factHints = ["is ", "are ", "uses ", "requires ", "supports ", "means ", "等于", "是"];
  const facts: DistilledKnowledge["facts"] = [];
  for (const chunk of chunks) {
    if (chunk.role !== "assistant" && chunk.role !== "tool") {
      continue;
    }
    const normalized = chunk.content.replace(/\s+/g, " ").trim();
    if (!normalized) {
      continue;
    }
    const lower = normalized.toLowerCase();
    if (!factHints.some((hint) => lower.includes(hint.toLowerCase()))) {
      continue;
    }
    facts.push({
      content: truncate(normalized, 220),
      kind: "fact",
      confidence: 0.62,
    });
    if (facts.length >= 30) {
      break;
    }
  }
  return facts;
}

function extractPreferences(chunks: DistillInputChunk[]): DistilledKnowledge["preferences"] {
  const prefHints = [
    "i prefer",
    "prefer ",
    "please always",
    "never ",
    "do not",
    "我喜欢",
    "我偏好",
    "请始终",
    "不要",
  ];
  const preferences: DistilledKnowledge["preferences"] = [];
  for (const chunk of chunks) {
    if (chunk.role !== "user") {
      continue;
    }
    const normalized = chunk.content.replace(/\s+/g, " ").trim();
    const lower = normalized.toLowerCase();
    if (!prefHints.some((hint) => lower.includes(hint.toLowerCase()))) {
      continue;
    }
    preferences.push({
      content: truncate(normalized, 200),
      confidence: 0.74,
    });
    if (preferences.length >= 20) {
      break;
    }
  }
  return preferences;
}

function extractLessons(chunks: DistillInputChunk[]): DistilledKnowledge["lessons"] {
  const lessonHints = [
    "lesson",
    "learned",
    "next time",
    "should have",
    "postmortem",
    "经验",
    "教训",
    "下次",
  ];
  const lessons: DistilledKnowledge["lessons"] = [];
  for (const chunk of chunks) {
    const normalized = chunk.content.replace(/\s+/g, " ").trim();
    const lower = normalized.toLowerCase();
    if (!lessonHints.some((hint) => lower.includes(hint.toLowerCase()))) {
      continue;
    }
    lessons.push({
      content: truncate(normalized, 240),
      context: `sequence:${chunk.sequence}`,
    });
    if (lessons.length >= 20) {
      break;
    }
  }
  return lessons;
}

function extractSkills(chunks: DistillInputChunk[], taskId: string): DistilledKnowledge["skills"] {
  const stepCandidates = chunks
    .filter((chunk) => chunk.role === "assistant" || chunk.role === "tool")
    .map((chunk) => chunk.content.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .filter((line) => /\b(run|create|update|test|deploy|verify|open|edit|fix|check)\b/i.test(line))
    .slice(0, 8)
    .map((line) => truncate(line, 140));

  if (stepCandidates.length < 3) {
    return [];
  }

  return [
    {
      name: `task-${taskId}-workflow`,
      description: "Reusable workflow extracted from task execution history.",
      steps: stepCandidates,
    },
  ];
}

function dedupeDistilledKnowledge(input: DistilledKnowledge): DistilledKnowledge {
  const dedupeByContent = <T extends { content: string }>(items: T[]) => {
    const seen = new Set<string>();
    const out: T[] = [];
    for (const item of items) {
      const normalized = normalizeText(item.content);
      if (!normalized || seen.has(normalized)) {
        continue;
      }
      seen.add(normalized);
      out.push(item);
    }
    return out;
  };

  const dedupeSkills = (skills: DistilledKnowledge["skills"]) => {
    const seen = new Set<string>();
    const out: DistilledKnowledge["skills"] = [];
    for (const skill of skills) {
      const normalized = normalizeText(skill.name + ":" + skill.steps.join("|"));
      if (!normalized || seen.has(normalized)) {
        continue;
      }
      seen.add(normalized);
      out.push({
        ...skill,
        steps: dedupeOrdered(skill.steps.map((step) => truncate(step, 180))),
      });
    }
    return out;
  };

  return {
    facts: dedupeByContent(input.facts).slice(0, 50),
    preferences: dedupeByContent(input.preferences).slice(0, 30),
    lessons: dedupeByContent(input.lessons).slice(0, 30),
    skills: dedupeSkills(input.skills).slice(0, 15),
  };
}

function normalizeText(input: string): string {
  return input.toLowerCase().replace(/\s+/g, " ").trim();
}

function dedupeOrdered(values: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const normalized = normalizeText(value);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    out.push(value);
  }
  return out;
}

function truncate(input: string, max: number): string {
  if (input.length <= max) {
    return input;
  }
  return `${input.slice(0, max - 3)}...`;
}

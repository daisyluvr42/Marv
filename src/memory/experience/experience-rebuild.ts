import type { MarvConfig } from "../../core/config/config.js";
import type { ExperienceConfig } from "../../core/config/types.memory.js";
import {
  readExperienceFile,
  writeExperienceFile,
  appendExperienceLog,
  EXPERIENCE_BUDGET_CHARS,
} from "./experience-files.js";

export type CalibrationResult = {
  driftDetected: boolean;
  driftReport?: string;
  zombieRemoved: string[];
  harmfulFlagged: string[];
  coreConfirmed: string[];
};

const CALIBRATION_SYSTEM_PROMPT = `You are an experience calibration engine. Your task is to ensure the experience document is accurate, effective, and not drifting.

Input:
1. Current EXPERIENCE.md (with a/p attribution markers)
2. Complete EXPERIENCE_LOG.md (audit trail)
3. Recent P3 episodic fragments (last 30 days)

Calibration rules:
- Check if the experience document has progressive drift from consecutive distillation biases
- Use the LOG as ground truth, compare with current document
- If needed, extract important missed experiences from P3 fragments

[Attribution-driven culling rules]:
- "Zombie experience": a:0 and exists > 30 days → remove (never activated, not relevant to actual work)
- "Harmful experience": a ≥ 5 and p/a < 0.3 → flag for review, analyze failure reasons, correct or remove
- "Core experience": a ≥ 10 and p/a > 0.7 → mark as core, prioritize retention
- "New experience": a < 3 → retain for observation, do not cull

Output:
DRIFT_DETECTED: true/false
DRIFT_REPORT: [if drift detected, describe the deviation and correction]
ZOMBIE_REMOVED: [list of removed zombie experiences]
HARMFUL_FLAGGED: [list of flagged harmful experiences]
CORE_CONFIRMED: [list of confirmed core experiences]
---
[Calibrated complete experience document]`;

/**
 * Run weekly calibration on EXPERIENCE.md.
 * Uses the highest-tier model for deep reasoning about experience quality.
 */
export async function weeklyCalibration(params: {
  agentId: string;
  cfg?: MarvConfig;
  p3Fragments?: string;
}): Promise<CalibrationResult> {
  const experienceConfig = resolveExperienceConfig(params.cfg);
  const experience = await readExperienceFile(params.agentId, "MARV_EXPERIENCE.md");
  const log = await readExperienceFile(params.agentId, "MARV_EXPERIENCE_LOG.md");

  if (!experience.trim() && !log.trim()) {
    return {
      driftDetected: false,
      zombieRemoved: [],
      harmfulFlagged: [],
      coreConfirmed: [],
    };
  }

  // Try LLM calibration
  try {
    const { inferLocal } = await import("../storage/local-llm-client.js");
    const modelConfig = params.cfg?.memory?.soul?.deepConsolidation?.model;

    const userPrompt = [
      "## Current EXPERIENCE.md",
      experience || "(empty)",
      "",
      "## EXPERIENCE_LOG.md",
      log || "(empty)",
      "",
      "## Recent P3 Episodic Fragments",
      params.p3Fragments || "(none available)",
    ].join("\n");

    const result = await inferLocal({
      cfg: params.cfg ?? {},
      model: modelConfig,
      system: CALIBRATION_SYSTEM_PROMPT,
      prompt: userPrompt,
    });

    if (result.ok) {
      const parsed = parseCalibrationOutput(result.text);
      const budget = experienceConfig.experienceBudgetChars ?? EXPERIENCE_BUDGET_CHARS;

      if (parsed.driftDetected && parsed.correctedExperience) {
        let corrected = parsed.correctedExperience;
        if (corrected.length > budget) {
          corrected = corrected.slice(0, budget);
        }
        await writeExperienceFile(params.agentId, "MARV_EXPERIENCE.md", corrected);
        await appendExperienceLog(
          params.agentId,
          `[${new Date().toISOString().slice(0, 16).replace("T", " ")}] [weekly_calibration] drift corrected | ${parsed.driftReport ?? ""}`,
        );
      }

      return {
        driftDetected: parsed.driftDetected,
        driftReport: parsed.driftReport,
        zombieRemoved: parsed.zombieRemoved,
        harmfulFlagged: parsed.harmfulFlagged,
        coreConfirmed: parsed.coreConfirmed,
      };
    }
  } catch {
    // LLM not available
  }

  // Heuristic fallback: run simple attribution-based culling
  return heuristicCalibration(params.agentId, experience, experienceConfig);
}

// --- Internal helpers ---

type ParsedCalibrationOutput = {
  driftDetected: boolean;
  driftReport?: string;
  correctedExperience?: string;
  zombieRemoved: string[];
  harmfulFlagged: string[];
  coreConfirmed: string[];
};

function parseCalibrationOutput(text: string): ParsedCalibrationOutput {
  const lines = text.split("\n");

  const driftLine = lines.find((l) => l.startsWith("DRIFT_DETECTED:"));
  const driftDetected = driftLine?.includes("true") ?? false;

  const driftReportLine = lines.find((l) => l.startsWith("DRIFT_REPORT:"));
  const driftReport = driftReportLine?.replace("DRIFT_REPORT:", "").trim();

  const zombieLine = lines.find((l) => l.startsWith("ZOMBIE_REMOVED:"));
  const zombieRemoved = parseListField(zombieLine);

  const harmfulLine = lines.find((l) => l.startsWith("HARMFUL_FLAGGED:"));
  const harmfulFlagged = parseListField(harmfulLine);

  const coreLine = lines.find((l) => l.startsWith("CORE_CONFIRMED:"));
  const coreConfirmed = parseListField(coreLine);

  // Extract corrected experience after ---
  const separatorIdx = text.indexOf("---");
  const correctedExperience = separatorIdx >= 0 ? text.slice(separatorIdx + 3).trim() : undefined;

  return {
    driftDetected,
    driftReport,
    correctedExperience,
    zombieRemoved,
    harmfulFlagged,
    coreConfirmed,
  };
}

function parseListField(line?: string): string[] {
  if (!line) {
    return [];
  }
  const content = line.replace(/^[A-Z_]+:/, "").trim();
  if (content === "[]" || content === "none" || !content) {
    return [];
  }
  return content
    .replace(/^\[/, "")
    .replace(/\]$/, "")
    .split(",")
    .map((s) => s.trim().replace(/^["']|["']$/g, ""))
    .filter(Boolean);
}

/**
 * Simple heuristic-based calibration when no LLM is available.
 * Removes zombie experiences (a:0 older than threshold) using regex.
 */
async function heuristicCalibration(
  agentId: string,
  experience: string,
  config: Partial<ExperienceConfig>,
): Promise<CalibrationResult> {
  const zombieAgeDays = config.zombieAgeDays ?? 30;
  const harmfulRatio = config.harmfulRatioThreshold ?? 0.3;

  const lines = experience.split("\n");
  const zombieRemoved: string[] = [];
  const harmfulFlagged: string[] = [];
  const coreConfirmed: string[] = [];
  const kept: string[] = [];

  const now = new Date();
  const datePattern = /@(\d{4}-\d{2})\s*\|\s*a:(\d+)\s+p:(\d+)/;

  for (const line of lines) {
    const match = datePattern.exec(line);
    if (!match) {
      kept.push(line);
      continue;
    }

    const [, dateStr, aStr, pStr] = match;
    const a = parseInt(aStr ?? "0", 10);
    const p = parseInt(pStr ?? "0", 10);
    const entryDate = new Date(`${dateStr}-01`);
    const ageDays = (now.getTime() - entryDate.getTime()) / (24 * 60 * 60 * 1000);

    // Zombie: a:0 and old
    if (a === 0 && ageDays > zombieAgeDays) {
      zombieRemoved.push(line.trim().slice(0, 80));
      continue;
    }

    // Harmful: high activation but low success rate
    if (a >= 5 && p / a < harmfulRatio) {
      harmfulFlagged.push(line.trim().slice(0, 80));
      // Keep but flag (don't remove in heuristic mode)
      kept.push(line);
      continue;
    }

    // Core: high activation and high success rate
    if (a >= 10 && p / a > 0.7) {
      coreConfirmed.push(line.trim().slice(0, 80));
    }

    kept.push(line);
  }

  const driftDetected = zombieRemoved.length > 0;
  if (driftDetected) {
    const updated = kept.join("\n");
    await writeExperienceFile(agentId, "MARV_EXPERIENCE.md", updated);
    await appendExperienceLog(
      agentId,
      `[${now.toISOString().slice(0, 16).replace("T", " ")}] [weekly_calibration_heuristic] removed ${zombieRemoved.length} zombies`,
    );
  }

  return {
    driftDetected,
    driftReport: driftDetected ? `Removed ${zombieRemoved.length} zombie experiences` : undefined,
    zombieRemoved,
    harmfulFlagged,
    coreConfirmed,
  };
}

function resolveExperienceConfig(cfg?: MarvConfig): Partial<ExperienceConfig> {
  return cfg?.memory?.experience ?? {};
}

import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../core/config/paths.js";
import type { ExperimentIteration, ExperimentState } from "./types.js";

/**
 * Write a human-readable Markdown experiment log.
 * Stored at: ~/.marv/state/proactive/{agentId}/experiments/{experimentId}/log.md
 */
export async function writeExperimentLog(agentId: string, state: ExperimentState): Promise<string> {
  const logDir = resolveExperimentLogDir(agentId, state.spec.id);
  await fs.mkdir(logDir, { recursive: true });
  const logPath = path.join(logDir, "log.md");
  const content = renderExperimentLog(state);
  await fs.writeFile(logPath, content, "utf-8");
  return logPath;
}

/** Resolve the directory for experiment logs. */
function resolveExperimentLogDir(agentId: string, experimentId: string): string {
  return path.join(resolveStateDir(process.env), "proactive", agentId, "experiments", experimentId);
}

/** Render the experiment log as Markdown. */
export function renderExperimentLog(state: ExperimentState): string {
  const lines: string[] = [];

  lines.push(`# Experiment: ${state.spec.name}`);
  lines.push(``);
  lines.push(`- **ID:** ${state.spec.id}`);
  lines.push(`- **Status:** ${state.status}`);
  if (state.stopReason) {
    lines.push(`- **Stop reason:** ${state.stopReason}`);
  }
  lines.push(`- **Objective:** ${state.spec.objective}`);
  lines.push(``);

  // Evaluators
  lines.push(`## Evaluators`);
  lines.push(``);
  for (const ev of state.spec.evaluators) {
    lines.push(`### ${ev.name}`);
    lines.push(`- Command: \`${ev.measureCommand}\``);
    lines.push(`- Direction: ${ev.direction}`);
    if (ev.threshold != null) {
      lines.push(`- Threshold: ${ev.threshold}`);
    }
    if (ev.minImprovementRatio != null) {
      lines.push(`- Min improvement: ${(ev.minImprovementRatio * 100).toFixed(1)}%`);
    }
    lines.push(``);
  }

  // Baseline
  if (state.iterations.length > 0) {
    const baseline = state.iterations[0].baseline;
    lines.push(`## Baseline`);
    lines.push(``);
    for (const r of baseline) {
      lines.push(`- **${r.evaluatorId}:** ${r.value}`);
    }
    lines.push(``);
  }

  // Iterations
  lines.push(`## Iterations`);
  lines.push(``);

  for (const iter of state.iterations) {
    const verdictEmoji = renderVerdict(iter.verdict);
    lines.push(`### Iteration ${iter.index} ${verdictEmoji}`);
    lines.push(``);

    if (iter.agentSummary) {
      lines.push(`**Change:** ${iter.agentSummary}`);
      lines.push(``);
    }

    if (iter.candidate) {
      for (const r of iter.candidate) {
        const base = iter.baseline.find((b) => b.evaluatorId === r.evaluatorId);
        const delta = base ? renderDelta(base.value, r.value) : "";
        lines.push(`- **${r.evaluatorId}:** ${r.value}${delta}`);
      }
    } else {
      lines.push(`- *(no measurement — error)*`);
    }

    lines.push(`- Tokens: ${iter.tokensUsed.toLocaleString()}`);
    lines.push(`- Duration: ${formatDuration(iter.durationMs)}`);
    lines.push(``);
  }

  // Summary
  lines.push(`## Summary`);
  lines.push(``);

  const kept = state.iterations.filter(
    (i) => i.verdict === "improved" || i.verdict === "threshold_met",
  ).length;
  const rolled = state.iterations.filter(
    (i) => i.verdict === "regressed" || i.verdict === "no_change",
  ).length;
  const errored = state.iterations.filter((i) => i.verdict === "error").length;

  if (state.bestResult) {
    lines.push(`**Best result** (iteration ${state.bestIteration ?? "N/A"}):`);
    for (const r of state.bestResult) {
      lines.push(`- **${r.evaluatorId}:** ${r.value}`);
    }
    lines.push(``);
  }

  lines.push(
    `- Total iterations: ${state.iterations.length} (${kept} kept, ${rolled} rolled back, ${errored} errors)`,
  );
  lines.push(`- Total tokens: ${state.totalTokensUsed.toLocaleString()}`);
  lines.push(`- Duration: ${formatDuration((state.completedAt ?? Date.now()) - state.startedAt)}`);
  lines.push(``);

  return lines.join("\n");
}

// ── Helpers ─────────────────────────────────────────────────────────

function renderVerdict(verdict: ExperimentIteration["verdict"]): string {
  switch (verdict) {
    case "improved":
      return "(kept)";
    case "threshold_met":
      return "(threshold met)";
    case "regressed":
      return "(rolled back)";
    case "no_change":
      return "(no change, rolled back)";
    case "error":
      return "(error)";
  }
}

function renderDelta(baseline: number, candidate: number): string {
  const diff = candidate - baseline;
  if (diff === 0) {
    return "";
  }
  const sign = diff > 0 ? "+" : "";
  const pct = baseline !== 0 ? ` / ${sign}${((diff / Math.abs(baseline)) * 100).toFixed(1)}%` : "";
  return ` (${sign}${diff.toFixed(4)}${pct})`;
}

function formatDuration(ms: number): string {
  if (ms > 60_000) {
    return `${(ms / 60_000).toFixed(1)}m`;
  }
  return `${(ms / 1_000).toFixed(1)}s`;
}

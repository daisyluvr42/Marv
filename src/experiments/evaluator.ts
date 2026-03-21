import { exec } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import type { EvaluatorResult, EvaluatorSpec, ExperimentVerdict, MetricParser } from "./types.js";

// ── Metric Parsing ──────────────────────────────────────────────────

/** Extract a numeric value from raw command output using the given parser. */
export function parseMetric(raw: string, parser: MetricParser): number {
  if (parser === "first_number") {
    const match = raw.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/i);
    if (!match) {
      return Number.NaN;
    }
    return Number(match[0]);
  }

  if (parser === "last_number") {
    const matches = [...raw.matchAll(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi)];
    if (matches.length === 0) {
      return Number.NaN;
    }
    return Number(matches[matches.length - 1][0]);
  }

  // Treat as regex with capture group
  const regex = new RegExp(parser);
  const match = raw.match(regex);
  if (!match) {
    return Number.NaN;
  }
  // Use first capture group if available, otherwise full match
  const value = match[1] ?? match[0];
  return Number(value);
}

// ── Evaluator Execution ─────────────────────────────────────────────

type RunEvaluatorOptions = {
  cwd?: string;
  env?: Record<string, string>;
};

/**
 * Run a single evaluator: execute its measure command, parse the metric.
 * In LLM-as-Judge mode (judgePrompt + judgeFile set), reads the file
 * and pipes content + scoring prompt to measureCommand via stdin.
 */
export async function runEvaluator(
  spec: EvaluatorSpec,
  opts: RunEvaluatorOptions = {},
): Promise<EvaluatorResult> {
  const cwd = spec.cwd ?? opts.cwd ?? process.cwd();
  const timeoutMs = (spec.timeoutSeconds ?? 300) * 1_000;
  const start = Date.now();

  try {
    let command = spec.measureCommand;

    // LLM-as-Judge mode: read file, construct stdin prompt, pipe to command
    if (spec.judgePrompt && spec.judgeFile) {
      const filePath = spec.judgeFile.startsWith("/")
        ? spec.judgeFile
        : path.resolve(cwd, spec.judgeFile);
      const fileContent = await fs.readFile(filePath, "utf-8");
      const stdinPrompt = buildJudgeStdin(spec.judgePrompt, fileContent);
      // Pipe the prompt to the measure command via stdin using a heredoc
      command = `cat <<'__JUDGE_EOF__' | ${spec.measureCommand}\n${stdinPrompt}\n__JUDGE_EOF__`;
    }

    const raw = await execCommand(command, { cwd, timeoutMs, env: opts.env });
    const value = parseMetric(raw, spec.metricParser);
    const durationMs = Date.now() - start;

    if (Number.isNaN(value)) {
      return {
        evaluatorId: spec.id,
        value: Number.NaN,
        raw,
        measuredAt: start,
        durationMs,
        error: `Failed to parse metric from output using parser "${spec.metricParser}"`,
      };
    }

    return { evaluatorId: spec.id, value, raw, measuredAt: start, durationMs };
  } catch (err) {
    return {
      evaluatorId: spec.id,
      value: Number.NaN,
      raw: "",
      measuredAt: start,
      durationMs: Date.now() - start,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

/** Build the stdin content for an LLM-as-Judge evaluation. */
function buildJudgeStdin(judgePrompt: string, fileContent: string): string {
  return [
    judgePrompt,
    "",
    "---",
    "Content to evaluate:",
    "---",
    fileContent,
    "---",
    "",
    "Respond with ONLY a single number (your score). No explanation.",
  ].join("\n");
}

/** Run all evaluators and return results. */
export async function runAllEvaluators(
  specs: EvaluatorSpec[],
  opts: RunEvaluatorOptions = {},
): Promise<EvaluatorResult[]> {
  // Run sequentially to avoid resource contention
  const results: EvaluatorResult[] = [];
  for (const spec of specs) {
    results.push(await runEvaluator(spec, opts));
  }
  return results;
}

// ── Result Comparison ───────────────────────────────────────────────

/** Compare a single evaluator's baseline vs candidate result. */
export function compareResults(
  baseline: EvaluatorResult,
  candidate: EvaluatorResult,
  spec: EvaluatorSpec,
): ExperimentVerdict {
  if (candidate.error || Number.isNaN(candidate.value)) {
    return "error";
  }
  if (baseline.error || Number.isNaN(baseline.value)) {
    return "error";
  }

  // Check absolute threshold first
  if (spec.threshold != null) {
    const met =
      spec.direction === "higher_is_better"
        ? candidate.value >= spec.threshold
        : candidate.value <= spec.threshold;
    if (met) {
      return "threshold_met";
    }
  }

  const diff = candidate.value - baseline.value;
  const absDiff = Math.abs(diff);

  // Check minimum improvement ratio
  const minRatio = spec.minImprovementRatio ?? 0;
  const baselineAbs = Math.abs(baseline.value);
  const ratio = baselineAbs > 0 ? absDiff / baselineAbs : absDiff > 0 ? Infinity : 0;

  if (ratio < minRatio) {
    return "no_change";
  }

  if (spec.direction === "higher_is_better") {
    return diff > 0 ? "improved" : diff < 0 ? "regressed" : "no_change";
  }
  // lower_is_better
  return diff < 0 ? "improved" : diff > 0 ? "regressed" : "no_change";
}

/**
 * Compare all evaluators. Verdicts:
 * - "threshold_met" if ANY evaluator hit its threshold and none regressed
 * - "improved" if ALL evaluators improved (or threshold_met) and none regressed
 * - "regressed" if ANY evaluator regressed
 * - "error" if ANY evaluator errored
 * - "no_change" otherwise
 */
export function compareAllResults(
  baselines: EvaluatorResult[],
  candidates: EvaluatorResult[],
  specs: EvaluatorSpec[],
): ExperimentVerdict {
  if (specs.length === 0) {
    return "error";
  }

  const verdicts: ExperimentVerdict[] = specs.map((spec, i) => {
    const baseline = baselines.find((r) => r.evaluatorId === spec.id) ?? baselines[i];
    const candidate = candidates.find((r) => r.evaluatorId === spec.id) ?? candidates[i];
    return compareResults(baseline, candidate, spec);
  });

  if (verdicts.includes("error")) {
    return "error";
  }
  if (verdicts.includes("regressed")) {
    return "regressed";
  }
  if (verdicts.every((v) => v === "threshold_met" || v === "improved")) {
    return verdicts.includes("threshold_met") ? "threshold_met" : "improved";
  }
  if (
    verdicts.some((v) => v === "threshold_met" || v === "improved") &&
    !verdicts.includes("regressed")
  ) {
    // Some improved, some no_change — count as improved overall
    return verdicts.includes("threshold_met") ? "threshold_met" : "improved";
  }
  return "no_change";
}

// ── Shell Execution Helper ──────────────────────────────────────────

function execCommand(
  command: string,
  opts: { cwd: string; timeoutMs: number; env?: Record<string, string> },
): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = exec(
      command,
      {
        cwd: opts.cwd,
        timeout: opts.timeoutMs,
        env: { ...process.env, ...opts.env },
        maxBuffer: 10 * 1024 * 1024, // 10 MB
        shell: "/bin/bash",
      },
      (error, stdout, stderr) => {
        if (error) {
          // Include stderr in the error message for debugging
          const msg = stderr ? `${error.message}\nstderr: ${stderr}` : error.message;
          reject(new Error(msg));
          return;
        }
        // Combine stdout and stderr — metric might be in either
        resolve(stdout + (stderr ? `\n${stderr}` : ""));
      },
    );

    // Safety: kill on timeout (exec timeout sends SIGTERM, but we add SIGKILL fallback)
    if (opts.timeoutMs > 0) {
      setTimeout(() => {
        child.kill("SIGKILL");
      }, opts.timeoutMs + 5_000);
    }
  });
}

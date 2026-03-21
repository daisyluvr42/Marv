import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { compareAllResults, compareResults, parseMetric, runEvaluator } from "./evaluator.js";
import type { EvaluatorResult, EvaluatorSpec } from "./types.js";

// ── parseMetric ─────────────────────────────────────────────────────

describe("parseMetric", () => {
  describe("first_number", () => {
    it("extracts the first number", () => {
      expect(parseMetric("coverage: 72.3% of lines", "first_number")).toBe(72.3);
    });

    it("handles integers", () => {
      expect(parseMetric("Found 42 issues", "first_number")).toBe(42);
    });

    it("handles negative numbers", () => {
      expect(parseMetric("delta: -3.5ms", "first_number")).toBe(-3.5);
    });

    it("handles scientific notation", () => {
      expect(parseMetric("loss: 1.5e-4", "first_number")).toBeCloseTo(0.00015);
    });

    it("returns NaN for no numbers", () => {
      expect(parseMetric("no numbers here", "first_number")).toBeNaN();
    });

    it("returns NaN for empty string", () => {
      expect(parseMetric("", "first_number")).toBeNaN();
    });
  });

  describe("last_number", () => {
    it("extracts the last number", () => {
      expect(parseMetric("step 100: loss=0.532, acc=0.891", "last_number")).toBe(0.891);
    });

    it("handles single number", () => {
      expect(parseMetric("result: 42", "last_number")).toBe(42);
    });

    it("returns NaN for no numbers", () => {
      expect(parseMetric("", "last_number")).toBeNaN();
    });
  });

  describe("regex parser", () => {
    it("extracts with capture group", () => {
      expect(parseMetric("coverage: 72.3% of lines", "(\\d+\\.?\\d*)%")).toBe(72.3);
    });

    it("falls back to full match without capture group", () => {
      expect(parseMetric("latency: 42ms", "\\d+")).toBe(42);
    });

    it("returns NaN when regex doesn't match", () => {
      expect(parseMetric("no match here", "(\\d+)%")).toBeNaN();
    });
  });
});

// ── compareResults ──────────────────────────────────────────────────

describe("compareResults", () => {
  const makeResult = (id: string, value: number, error?: string): EvaluatorResult => ({
    evaluatorId: id,
    value,
    raw: String(value),
    measuredAt: Date.now(),
    durationMs: 100,
    error,
  });

  const makeSpec = (overrides: Partial<EvaluatorSpec> = {}): EvaluatorSpec => ({
    id: "test",
    name: "test",
    measureCommand: "echo 1",
    metricParser: "first_number",
    direction: "higher_is_better",
    ...overrides,
  });

  it("returns improved when candidate is higher (higher_is_better)", () => {
    const spec = makeSpec({ direction: "higher_is_better" });
    expect(compareResults(makeResult("test", 70), makeResult("test", 75), spec)).toBe("improved");
  });

  it("returns regressed when candidate is lower (higher_is_better)", () => {
    const spec = makeSpec({ direction: "higher_is_better" });
    expect(compareResults(makeResult("test", 70), makeResult("test", 65), spec)).toBe("regressed");
  });

  it("returns improved when candidate is lower (lower_is_better)", () => {
    const spec = makeSpec({ direction: "lower_is_better" });
    expect(compareResults(makeResult("test", 1.0), makeResult("test", 0.8), spec)).toBe("improved");
  });

  it("returns regressed when candidate is higher (lower_is_better)", () => {
    const spec = makeSpec({ direction: "lower_is_better" });
    expect(compareResults(makeResult("test", 1.0), makeResult("test", 1.2), spec)).toBe(
      "regressed",
    );
  });

  it("returns no_change when values are equal", () => {
    const spec = makeSpec({ direction: "higher_is_better" });
    expect(compareResults(makeResult("test", 70), makeResult("test", 70), spec)).toBe("no_change");
  });

  it("returns threshold_met when threshold is crossed (higher_is_better)", () => {
    const spec = makeSpec({ direction: "higher_is_better", threshold: 80 });
    expect(compareResults(makeResult("test", 70), makeResult("test", 85), spec)).toBe(
      "threshold_met",
    );
  });

  it("returns threshold_met when threshold is crossed (lower_is_better)", () => {
    const spec = makeSpec({ direction: "lower_is_better", threshold: 0.5 });
    expect(compareResults(makeResult("test", 1.0), makeResult("test", 0.4), spec)).toBe(
      "threshold_met",
    );
  });

  it("returns regressed even when threshold exists but not met and value is worse", () => {
    const spec = makeSpec({ direction: "higher_is_better", threshold: 90 });
    expect(compareResults(makeResult("test", 70), makeResult("test", 65), spec)).toBe("regressed");
  });

  it("returns no_change when improvement is below minImprovementRatio", () => {
    const spec = makeSpec({ direction: "higher_is_better", minImprovementRatio: 0.05 });
    // 70 → 71 = 1.4% improvement, below 5% threshold
    expect(compareResults(makeResult("test", 70), makeResult("test", 71), spec)).toBe("no_change");
  });

  it("returns improved when improvement meets minImprovementRatio", () => {
    const spec = makeSpec({ direction: "higher_is_better", minImprovementRatio: 0.05 });
    // 70 → 74 = 5.7% improvement, above 5% threshold
    expect(compareResults(makeResult("test", 70), makeResult("test", 74), spec)).toBe("improved");
  });

  it("returns error when candidate has error", () => {
    const spec = makeSpec();
    expect(
      compareResults(makeResult("test", 70), makeResult("test", Number.NaN, "timeout"), spec),
    ).toBe("error");
  });

  it("returns error when baseline has error", () => {
    const spec = makeSpec();
    expect(
      compareResults(makeResult("test", Number.NaN, "fail"), makeResult("test", 70), spec),
    ).toBe("error");
  });
});

// ── compareAllResults ───────────────────────────────────────────────

describe("compareAllResults", () => {
  const makeResult = (id: string, value: number): EvaluatorResult => ({
    evaluatorId: id,
    value,
    raw: String(value),
    measuredAt: Date.now(),
    durationMs: 100,
  });

  const makeSpec = (id: string, dir: "higher_is_better" | "lower_is_better"): EvaluatorSpec => ({
    id,
    name: id,
    measureCommand: `echo ${id}`,
    metricParser: "first_number",
    direction: dir,
  });

  it("returns improved when all evaluators improve", () => {
    const specs = [makeSpec("cov", "higher_is_better"), makeSpec("speed", "lower_is_better")];
    const baselines = [makeResult("cov", 70), makeResult("speed", 100)];
    const candidates = [makeResult("cov", 75), makeResult("speed", 90)];
    expect(compareAllResults(baselines, candidates, specs)).toBe("improved");
  });

  it("returns regressed when any evaluator regresses", () => {
    const specs = [makeSpec("cov", "higher_is_better"), makeSpec("speed", "lower_is_better")];
    const baselines = [makeResult("cov", 70), makeResult("speed", 100)];
    const candidates = [makeResult("cov", 75), makeResult("speed", 110)]; // speed regressed
    expect(compareAllResults(baselines, candidates, specs)).toBe("regressed");
  });

  it("returns improved when some improve and some stay the same", () => {
    const specs = [makeSpec("cov", "higher_is_better"), makeSpec("speed", "lower_is_better")];
    const baselines = [makeResult("cov", 70), makeResult("speed", 100)];
    const candidates = [makeResult("cov", 75), makeResult("speed", 100)]; // speed no_change
    expect(compareAllResults(baselines, candidates, specs)).toBe("improved");
  });

  it("returns no_change when all evaluators show no change", () => {
    const specs = [makeSpec("cov", "higher_is_better"), makeSpec("speed", "lower_is_better")];
    const baselines = [makeResult("cov", 70), makeResult("speed", 100)];
    const candidates = [makeResult("cov", 70), makeResult("speed", 100)];
    expect(compareAllResults(baselines, candidates, specs)).toBe("no_change");
  });

  it("returns threshold_met when any evaluator hits threshold and none regressed", () => {
    const specs = [
      { ...makeSpec("cov", "higher_is_better"), threshold: 80 },
      makeSpec("speed", "lower_is_better"),
    ];
    const baselines = [makeResult("cov", 70), makeResult("speed", 100)];
    const candidates = [makeResult("cov", 85), makeResult("speed", 95)];
    expect(compareAllResults(baselines, candidates, specs)).toBe("threshold_met");
  });

  it("returns error for empty specs", () => {
    expect(compareAllResults([], [], [])).toBe("error");
  });

  it("matches evaluators by id, not index", () => {
    const specs = [makeSpec("alpha", "higher_is_better"), makeSpec("beta", "lower_is_better")];
    // Reversed order in results
    const baselines = [makeResult("beta", 100), makeResult("alpha", 70)];
    const candidates = [makeResult("beta", 90), makeResult("alpha", 75)];
    expect(compareAllResults(baselines, candidates, specs)).toBe("improved");
  });
});

// ── runEvaluator ────────────────────────────────────────────────────

describe("runEvaluator", () => {
  it("runs a command and parses the metric", async () => {
    const spec: EvaluatorSpec = {
      id: "lines",
      name: "line count",
      measureCommand: 'echo "total: 42 lines"',
      metricParser: "first_number",
      direction: "lower_is_better",
      timeoutSeconds: 5,
    };
    const result = await runEvaluator(spec);
    expect(result.value).toBe(42);
    expect(result.error).toBeUndefined();
    expect(result.evaluatorId).toBe("lines");
  });

  it("returns error for failing command", async () => {
    const spec: EvaluatorSpec = {
      id: "fail",
      name: "failing",
      measureCommand: "exit 1",
      metricParser: "first_number",
      direction: "higher_is_better",
      timeoutSeconds: 5,
    };
    const result = await runEvaluator(spec);
    expect(result.error).toBeDefined();
    expect(result.value).toBeNaN();
  });

  it("returns error when metric cannot be parsed", async () => {
    const spec: EvaluatorSpec = {
      id: "nonum",
      name: "no number",
      measureCommand: 'echo "no numbers"',
      metricParser: "first_number",
      direction: "higher_is_better",
      timeoutSeconds: 5,
    };
    const result = await runEvaluator(spec);
    expect(result.value).toBeNaN();
    expect(result.error).toContain("Failed to parse metric");
  });

  it("respects timeout", async () => {
    const spec: EvaluatorSpec = {
      id: "slow",
      name: "slow",
      measureCommand: "sleep 60",
      metricParser: "first_number",
      direction: "higher_is_better",
      timeoutSeconds: 1,
    };
    const result = await runEvaluator(spec);
    expect(result.error).toBeDefined();
  }, 10_000);
});

// ── LLM-as-Judge mode ───────────────────────────────────────────────

describe("runEvaluator (LLM-as-Judge)", () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = path.join(os.tmpdir(), `marv-judge-test-${Date.now()}`);
    await fs.mkdir(tmpDir, { recursive: true });
  });

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
  });

  it("reads judgeFile and pipes content + judgePrompt to measureCommand", async () => {
    // Create a file to judge
    const testFile = path.join(tmpDir, "draft.txt");
    await fs.writeFile(testFile, "Buy our amazing product today!");

    const spec: EvaluatorSpec = {
      id: "judge",
      name: "Copy Judge",
      // The "LLM" in this test is just grep + wc that counts words and outputs a fixed score
      measureCommand: "wc -w | awk '{print 8}'",
      metricParser: "first_number",
      direction: "higher_is_better",
      judgePrompt: "Score this copy 1-10 on clarity.",
      judgeFile: testFile,
      timeoutSeconds: 5,
    };

    const result = await runEvaluator(spec);
    expect(result.value).toBe(8);
    expect(result.error).toBeUndefined();
  });

  it("handles relative judgeFile paths", async () => {
    await fs.writeFile(path.join(tmpDir, "content.md"), "Hello world");

    const spec: EvaluatorSpec = {
      id: "rel",
      name: "Relative Path Judge",
      measureCommand: "wc -w | awk '{print 7}'",
      metricParser: "first_number",
      direction: "higher_is_better",
      judgePrompt: "Rate this.",
      judgeFile: "content.md",
      timeoutSeconds: 5,
    };

    const result = await runEvaluator(spec, { cwd: tmpDir });
    expect(result.value).toBe(7);
  });

  it("returns error when judgeFile does not exist", async () => {
    const spec: EvaluatorSpec = {
      id: "missing",
      name: "Missing File",
      measureCommand: "cat",
      metricParser: "first_number",
      direction: "higher_is_better",
      judgePrompt: "Score this.",
      judgeFile: path.join(tmpDir, "nonexistent.txt"),
      timeoutSeconds: 5,
    };

    const result = await runEvaluator(spec);
    expect(result.error).toBeDefined();
    expect(result.value).toBeNaN();
  });

  it("includes file content in the stdin sent to measureCommand", async () => {
    const testFile = path.join(tmpDir, "data.txt");
    await fs.writeFile(testFile, "The quick brown fox");

    const spec: EvaluatorSpec = {
      id: "echo",
      name: "Echo Judge",
      // Just output word count of the piped content as the "score"
      measureCommand: "grep -c 'quick brown fox'",
      metricParser: "first_number",
      direction: "higher_is_better",
      judgePrompt: "Evaluate the following text.",
      judgeFile: testFile,
      timeoutSeconds: 5,
    };

    const result = await runEvaluator(spec);
    // grep -c counts lines matching the pattern — file content is in the piped input
    expect(result.value).toBe(1);
    expect(result.error).toBeUndefined();
  });
});

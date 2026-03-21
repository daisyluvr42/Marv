// ── Evaluator Types ──────────────────────────────────────────────────

export type MetricDirection = "higher_is_better" | "lower_is_better";

/**
 * Built-in metric parsers:
 * - "first_number": extracts the first numeric value from stdout
 * - "last_number": extracts the last numeric value from stdout
 * - Any other string is treated as a regex with a capture group, e.g. `/(\d+\.?\d*)%/`
 */
export type MetricParser = string;

export type EvaluatorSpec = {
  id: string;
  name: string;
  /**
   * Shell command to produce the metric. Runs in the experiment workspace.
   * For LLM-as-Judge: this is the LLM CLI command that reads stdin and outputs a response.
   * When `judgePrompt` is set, file content + prompt are piped to this command via stdin.
   */
  measureCommand: string;
  /** How to extract a numeric value from command output. */
  metricParser: MetricParser;
  /** Whether higher or lower metric values are better. */
  direction: MetricDirection;
  /** Absolute target: experiment succeeds early if metric crosses this. */
  threshold?: number;
  /** Minimum relative improvement ratio to count as "improved" (e.g. 0.01 = 1%). */
  minImprovementRatio?: number;
  /** Max seconds for the measure command before timeout. */
  timeoutSeconds?: number;
  /** Working directory for measurement (defaults to experiment workspace root). */
  cwd?: string;
  /**
   * LLM-as-Judge mode: scoring prompt to send to an LLM.
   * When set, the evaluator reads `judgeFile`, constructs a scoring prompt,
   * and pipes it to `measureCommand` via stdin.
   * The prompt should instruct the LLM to output a numeric score.
   */
  judgePrompt?: string;
  /**
   * File to read and send to the LLM judge for evaluation.
   * Required when `judgePrompt` is set.
   */
  judgeFile?: string;
};

export type EvaluatorResult = {
  evaluatorId: string;
  value: number;
  /** Full stdout from the measure command, for debugging. */
  raw: string;
  measuredAt: number;
  durationMs: number;
  error?: string;
};

// ── Experiment Verdict ──────────────────────────────────────────────

export type ExperimentVerdict =
  | "improved" // all evaluators show improvement
  | "regressed" // at least one evaluator got worse
  | "no_change" // within noise / improvement ratio threshold
  | "error" // measurement or mutation failed
  | "threshold_met"; // hit the absolute threshold, stop early

// ── Checkpoint Types ────────────────────────────────────────────────

export type CheckpointRef = {
  /** Which strategy produced this checkpoint. */
  strategy: string;
  /** Strategy-specific reference (commit hash, tmp path, snapshot key, etc.). */
  ref: string;
  metadata?: Record<string, unknown>;
};

/**
 * Pluggable checkpoint/rollback mechanism.
 * The experiment loop calls save() before each mutation and restore() on regression.
 */
export interface CheckpointStrategy {
  /** Create a checkpoint and return a ref that can be used to restore later. */
  save(label: string): Promise<CheckpointRef>;
  /** Restore state to a previously saved checkpoint. */
  restore(ref: CheckpointRef): Promise<void>;
  /** Human-readable description of this strategy (for logs). */
  describe(): string;
}

/** Configuration for selecting a checkpoint strategy. */
export type CheckpointConfig =
  | { strategy: "git"; cwd?: string }
  | { strategy: "file-copy"; paths: string[] }
  | { strategy: "json-snapshot"; snapshotDir?: string }
  | { strategy: "none" };

// ── Experiment Types ────────────────────────────────────────────────

export type ExperimentConstraints = {
  /** Glob patterns for files the agent may modify. Empty = unrestricted. */
  allowedFiles?: string[];
  /** Glob patterns for files the agent must not modify. */
  deniedFiles?: string[];
  /** Glob patterns for high-risk paths that require human confirmation before mutation. */
  dangerousPaths?: string[];
  /** Max total tokens across all iterations. */
  tokenBudget?: number;
  /** Max wall-clock time in seconds for the entire experiment. */
  timeBudgetSeconds?: number;
};

export type ExperimentSpec = {
  id: string;
  name: string;
  /** Evaluator(s) to measure progress. Multiple = all must pass. */
  evaluators: EvaluatorSpec[];
  /** What the agent should try to improve. Becomes the mutation prompt. */
  objective: string;
  /** Constraints on what the agent can modify. */
  constraints: ExperimentConstraints;
  /** Max iterations before giving up. */
  maxIterations: number;
  /** Checkpoint/rollback strategy. */
  checkpoint: CheckpointConfig;
  /** Associated proactive goal ID, if any. */
  goalId?: string;
};

export type ExperimentIteration = {
  index: number;
  baseline: EvaluatorResult[];
  candidate: EvaluatorResult[] | null;
  verdict: ExperimentVerdict;
  /** Checkpoint ref for rollback. */
  checkpointRef?: CheckpointRef;
  /** What the agent said it changed. */
  agentSummary?: string;
  tokensUsed: number;
  durationMs: number;
  startedAt: number;
};

export type ExperimentStatus = "pending" | "running" | "completed" | "failed" | "stopped";

export type ExperimentState = {
  spec: ExperimentSpec;
  status: ExperimentStatus;
  iterations: ExperimentIteration[];
  bestResult: EvaluatorResult[] | null;
  bestIteration: number | null;
  totalTokensUsed: number;
  startedAt: number;
  completedAt?: number;
  stopReason?: string;
};

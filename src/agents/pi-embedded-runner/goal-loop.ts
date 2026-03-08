import type {
  ToolCallRecord,
  ToolLoopEventRecord,
} from "../../logging/diagnostic-session-state.js";
import type { EmbeddedRunAttemptResult } from "./run/types.js";

export type GoalFrame = {
  objective: string;
  successCriteria: string[];
  constraints: string[];
  complexity: "trivial" | "moderate" | "complex";
  goalType: "inquiry" | "mutation";
};

export type ProblemShape =
  | "information_gap"
  | "implementation_blocked"
  | "validation_failure"
  | "tool_or_permission_limit"
  | "search_drift"
  | "parallelizable_subproblems";

export type StrategyFamily =
  | "read_context"
  | "inspect_failure"
  | "try_alternative"
  | "validate_result"
  | "request_capability"
  | "recenter_goal"
  | "split_subproblems"
  | "wrap_up";

export type ProgressSignal = {
  attemptIndex: number;
  newEvidence: boolean;
  errorStateChanged: boolean;
  resultDiversity: number;
  toolLoopEvent: "none" | "warning" | "critical";
  cleanCompletion: boolean;
  toolBreadth: number;
  deliveredOutput: boolean;
};

export type ProgressClassification = "advancing" | "ambiguous" | "stalled" | "completed";

export type StrategyHint = {
  memoryId: string;
  summary: string;
  strategyFamily: StrategyFamily | null;
  problemShape: ProblemShape | null;
  score: number;
  tier?: string;
};

export type GoalLoopGuardLevel = "normal" | "warning" | "force_shift" | "stop";

export type GoalLoopState = {
  goalFrame: GoalFrame;
  directionNodes: string[];
  currentNodeIndex: number;
  strategyFamily: StrategyFamily;
  problemShape: ProblemShape | null;
  progressWindow: ProgressSignal[];
  stuckCounter: number;
  shiftCount: number;
  loopGuardLevel: GoalLoopGuardLevel;
  convergeReason: string | null;
  priorStrategyHints: StrategyHint[];
  attemptCount: number;
  lastErrorFingerprint: string | null;
};

export type GoalProgressReview = {
  signal: ProgressSignal;
  classification: ProgressClassification;
  problemShape: ProblemShape | null;
  strategyFamily: StrategyFamily;
  steeringContext: string | null;
  visibility: string;
  state: GoalLoopState;
};

const TRIVIAL_PROMPT_MAX_CHARS = 48;
const READ_LIKE_TOOL_RE =
  /\b(read|view|open|search|grep|rg|find|inspect|list|ls|status|history|get)\b/i;
const WRITE_LIKE_TOOL_RE =
  /\b(write|edit|apply_patch|create|append|update|delete|rm|move|rename|replace|commit)\b/i;
const VALIDATION_TOOL_RE = /\b(test|lint|check|validate|verify|build|typecheck|coverage|smoke)\b/i;
const GREETING_RE =
  /^(hi|hello|hey|yo|thanks|thank you|ok|okay|cool|nice|good morning|good evening)[!. ]*$/i;
const SIMPLE_FACT_RE = /^(what('?s| is)|who('?s| is)|when is|where is|why is|how many)\b/i;
const MUTATION_RE =
  /\b(fix|implement|add|update|refactor|change|edit|create|remove|delete|rename|wire|support)\b/i;
const CONSTRAINT_RE =
  /\b(do not|don't|without|avoid|must|need to|should|keep|preserve|no brainstorm|不要|必须|保留)\b/i;
const PERMISSION_RE =
  /\b(permission|permissions|sandbox|approval|allow|forbidden|denied|escalat|missing tool|not allowed)\b/i;

function clampRatio(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function normalizeLine(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function toSentenceList(parts: string[], fallback: string[]): string[] {
  const normalized = parts.map(normalizeLine).filter(Boolean);
  if (normalized.length > 0) {
    return normalized.slice(0, 4);
  }
  return fallback;
}

function extractPromptConstraints(prompt: string): string[] {
  const lines = prompt.split(/\r?\n/).map(normalizeLine).filter(Boolean);
  const matches = lines.filter((line) => CONSTRAINT_RE.test(line));
  return toSentenceList(matches, []);
}

function resolveGoalComplexity(prompt: string): GoalFrame["complexity"] {
  const lowered = prompt.toLowerCase();
  if (prompt.length < 120 && !/\n/.test(prompt) && !MUTATION_RE.test(lowered)) {
    return "trivial";
  }
  if (
    prompt.length > 550 ||
    /```/.test(prompt) ||
    /\b(test|architecture|multi|several|across)\b/i.test(prompt)
  ) {
    return "complex";
  }
  return "moderate";
}

function resolveGoalType(prompt: string): GoalFrame["goalType"] {
  return MUTATION_RE.test(prompt) ? "mutation" : "inquiry";
}

function summarizeObjective(prompt: string): string {
  const firstMeaningful = prompt.split(/\r?\n/).map(normalizeLine).find(Boolean) ?? "";
  if (!firstMeaningful) {
    return "Complete the user's request.";
  }
  if (firstMeaningful.length <= 160) {
    return firstMeaningful;
  }
  return `${firstMeaningful.slice(0, 157)}...`;
}

function buildSuccessCriteria(goalType: GoalFrame["goalType"]): string[] {
  if (goalType === "mutation") {
    return [
      "Implement the requested change with the smallest useful scope.",
      "Validate the result when local checks are available.",
      "Report the outcome and any remaining limits succinctly.",
    ];
  }
  return [
    "Answer the user's question directly.",
    "Ground the answer in the gathered evidence.",
    "Call out uncertainty or limits when they remain.",
  ];
}

function buildDirectionNodes(goalType: GoalFrame["goalType"]): string[] {
  if (goalType === "mutation") {
    return [
      "understand current state",
      "identify the smallest safe change",
      "apply the targeted change",
      "validate the result",
      "wrap up",
    ];
  }
  return [
    "clarify the information need",
    "gather targeted evidence",
    "synthesize the answer",
    "verify key points",
    "wrap up",
  ];
}

export function shouldSkipGoalFrame(prompt: string): boolean {
  const trimmed = prompt.trim();
  if (!trimmed) {
    return true;
  }
  if (trimmed.length <= TRIVIAL_PROMPT_MAX_CHARS && GREETING_RE.test(trimmed)) {
    return true;
  }
  if (
    trimmed.length <= 72 &&
    SIMPLE_FACT_RE.test(trimmed) &&
    !MUTATION_RE.test(trimmed) &&
    !/[/\\]/.test(trimmed)
  ) {
    return true;
  }
  return false;
}

export function generateHeuristicGoalFrame(prompt: string): {
  goalFrame: GoalFrame;
  directionNodes: string[];
} {
  const goalType = resolveGoalType(prompt);
  return {
    goalFrame: {
      objective: summarizeObjective(prompt),
      successCriteria: buildSuccessCriteria(goalType),
      constraints: extractPromptConstraints(prompt),
      complexity: resolveGoalComplexity(prompt),
      goalType,
    },
    directionNodes: buildDirectionNodes(goalType),
  };
}

function resolveInitialStrategyFamily(goalType: GoalFrame["goalType"]): StrategyFamily {
  return goalType === "mutation" ? "read_context" : "read_context";
}

export function createGoalLoopState(args: {
  prompt: string;
  priorStrategyHints?: StrategyHint[];
}): GoalLoopState | null {
  if (shouldSkipGoalFrame(args.prompt)) {
    return null;
  }
  const { goalFrame, directionNodes } = generateHeuristicGoalFrame(args.prompt);
  return {
    goalFrame,
    directionNodes,
    currentNodeIndex: 0,
    strategyFamily: resolveInitialStrategyFamily(goalFrame.goalType),
    problemShape: null,
    progressWindow: [],
    stuckCounter: 0,
    shiftCount: 0,
    loopGuardLevel: "normal",
    convergeReason: null,
    priorStrategyHints: args.priorStrategyHints ?? [],
    attemptCount: 0,
    lastErrorFingerprint: null,
  };
}

function dedupeStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const normalized = normalizeLine(value);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

export function buildGoalSteeringContext(
  state: GoalLoopState,
  options?: { includeAnchor?: boolean },
): string | null {
  const currentNode =
    state.directionNodes[state.currentNodeIndex] ??
    state.directionNodes.at(-1) ??
    "make the next best move";
  const lines: string[] = [];
  if (options?.includeAnchor !== false) {
    lines.push("Goal focus:");
    lines.push(`Objective: ${state.goalFrame.objective}`);
    if (state.goalFrame.constraints.length > 0) {
      lines.push(`Constraints: ${state.goalFrame.constraints.join(" | ")}`);
    }
    lines.push(`Current node: ${currentNode}`);
  }
  if (state.priorStrategyHints.length > 0) {
    const hints = state.priorStrategyHints.slice(0, 2).map((hint) => `- ${hint.summary}`);
    lines.push("Relevant prior strategy hints:");
    lines.push(...hints);
  }

  const strategyFragment = buildStrategyPromptFragment(state);
  if (strategyFragment) {
    lines.push(strategyFragment);
  }
  const merged = lines.map(normalizeLine).filter(Boolean).join("\n");
  return merged || null;
}

export function buildStrategyPromptFragment(
  state: Pick<GoalLoopState, "goalFrame" | "strategyFamily" | "problemShape" | "loopGuardLevel">,
): string | null {
  if (state.loopGuardLevel === "force_shift") {
    return `Your current approach is not producing progress toward "${state.goalFrame.objective}". Try a fundamentally different strategy family and do not repeat the same category of actions.`;
  }
  if (state.loopGuardLevel === "warning") {
    return `Progress looks weak. Re-center on "${state.goalFrame.objective}", restate the blocker briefly, and choose the smallest direct move that can change the evidence.`;
  }
  if (state.strategyFamily === "inspect_failure") {
    return "The last attempt hit a blocker. Inspect the failure closely before retrying and compare expected versus actual behavior.";
  }
  if (state.strategyFamily === "validate_result") {
    return "Prefer validation over more edits. Run the narrowest useful check and use that result to decide the next move.";
  }
  if (state.strategyFamily === "request_capability") {
    return "You are near a capability boundary. First exhaust safe local alternatives, then request missing tools or escalation only if the path is still blocked.";
  }
  if (state.strategyFamily === "recenter_goal") {
    return `You appear to be exploring too broadly. Return to the core objective: "${state.goalFrame.objective}" and act on the most relevant blocker only.`;
  }
  if (state.strategyFamily === "split_subproblems") {
    return "Separate independent subproblems cleanly. Only delegate or parallelize when the work can proceed independently.";
  }
  if (state.problemShape === "information_gap") {
    return "Focus on gathering specific evidence before making changes. Read targeted files, outputs, or history that directly address the blocker.";
  }
  return null;
}

function resolveToolLoopLevel(events: ToolLoopEventRecord[]): ProgressSignal["toolLoopEvent"] {
  if (events.some((event) => event.level === "critical")) {
    return "critical";
  }
  if (events.some((event) => event.level === "warning")) {
    return "warning";
  }
  return "none";
}

function buildErrorFingerprint(
  attempt: EmbeddedRunAttemptResult,
  promptErrorText?: string,
): string | null {
  if (attempt.lastToolError) {
    return [
      normalizeLine(attempt.lastToolError.toolName),
      normalizeLine(attempt.lastToolError.meta ?? ""),
      normalizeLine(attempt.lastToolError.error ?? ""),
      attempt.lastToolError.actionFingerprint ?? "",
    ]
      .filter(Boolean)
      .join("|");
  }
  const assistantError =
    attempt.lastAssistant?.stopReason === "error"
      ? normalizeLine(attempt.lastAssistant.errorMessage ?? "")
      : "";
  const promptError = normalizeLine(promptErrorText ?? "");
  const merged = dedupeStrings([assistantError, promptError]).join("|");
  return merged || null;
}

export function deriveProgressSignal(args: {
  state: GoalLoopState;
  attempt: EmbeddedRunAttemptResult;
  recentToolCalls: ToolCallRecord[];
  priorResultHashes: Set<string>;
  recentLoopEvents: ToolLoopEventRecord[];
  promptErrorText?: string;
}): { signal: ProgressSignal; errorFingerprint: string | null } {
  const resultHashes = args.recentToolCalls
    .map((record) => record.resultHash)
    .filter((value): value is string => typeof value === "string" && value.length > 0);
  const uniqueResultHashes = new Set(resultHashes);
  const distinctTools = new Set(
    args.recentToolCalls.map((record) => record.toolName).filter(Boolean),
  );
  const cleanCompletion =
    !args.attempt.aborted &&
    !args.attempt.promptError &&
    args.attempt.lastAssistant?.stopReason !== "error";
  const deliveredOutput =
    args.attempt.didSendViaMessagingTool ||
    args.attempt.messagingToolSentTexts.length > 0 ||
    args.attempt.assistantTexts.some((text) => text.trim().length > 0);
  const errorFingerprint = buildErrorFingerprint(args.attempt, args.promptErrorText);
  const signal: ProgressSignal = {
    attemptIndex: args.state.attemptCount + 1,
    newEvidence: [...uniqueResultHashes].some((hash) => !args.priorResultHashes.has(hash)),
    errorStateChanged: errorFingerprint !== args.state.lastErrorFingerprint,
    resultDiversity: clampRatio(
      resultHashes.length > 0 ? uniqueResultHashes.size / resultHashes.length : 0,
    ),
    toolLoopEvent: resolveToolLoopLevel(args.recentLoopEvents),
    cleanCompletion,
    toolBreadth: distinctTools.size,
    deliveredOutput,
  };
  return { signal, errorFingerprint };
}

export function classifyProgress(signal: ProgressSignal): ProgressClassification {
  if (signal.cleanCompletion && signal.deliveredOutput) {
    return "completed";
  }
  if (
    signal.newEvidence &&
    (signal.errorStateChanged || signal.resultDiversity >= 0.5 || signal.toolBreadth >= 2)
  ) {
    return "advancing";
  }
  if (
    signal.toolLoopEvent === "critical" ||
    (!signal.newEvidence && signal.resultDiversity < 0.2) ||
    (!signal.errorStateChanged && signal.toolBreadth <= 1)
  ) {
    return "stalled";
  }
  return "ambiguous";
}

function recentToolNames(records: ToolCallRecord[]): string[] {
  return records.map((record) => record.toolName).filter(Boolean);
}

function hasRecentToolMatching(records: ToolCallRecord[], pattern: RegExp): boolean {
  return recentToolNames(records).some((toolName) => pattern.test(toolName));
}

function collectAttemptErrorText(
  attempt: EmbeddedRunAttemptResult,
  promptErrorText?: string,
): string {
  return dedupeStrings([
    attempt.lastToolError?.error ?? "",
    attempt.lastAssistant?.errorMessage ?? "",
    promptErrorText ?? "",
  ]).join(" ");
}

export function classifyProblemShape(args: {
  attempt: EmbeddedRunAttemptResult;
  recentToolCalls: ToolCallRecord[];
  signal: ProgressSignal;
  promptErrorText?: string;
}): ProblemShape | null {
  const errorText = collectAttemptErrorText(args.attempt, args.promptErrorText);
  if (PERMISSION_RE.test(errorText)) {
    return "tool_or_permission_limit";
  }
  if (
    hasRecentToolMatching(args.recentToolCalls, VALIDATION_TOOL_RE) &&
    (args.attempt.lastToolError || args.attempt.lastAssistant?.stopReason === "error")
  ) {
    return "validation_failure";
  }
  if (
    args.attempt.lastToolError?.mutatingAction ||
    (hasRecentToolMatching(args.recentToolCalls, WRITE_LIKE_TOOL_RE) &&
      Boolean(args.attempt.lastToolError || args.attempt.promptError))
  ) {
    return "implementation_blocked";
  }
  if (args.signal.toolBreadth >= 4 && args.signal.resultDiversity < 0.45) {
    return "search_drift";
  }
  if (args.signal.toolBreadth >= 4 && args.signal.resultDiversity >= 0.55) {
    return "parallelizable_subproblems";
  }
  if (
    hasRecentToolMatching(args.recentToolCalls, READ_LIKE_TOOL_RE) &&
    !hasRecentToolMatching(args.recentToolCalls, WRITE_LIKE_TOOL_RE)
  ) {
    return "information_gap";
  }
  return null;
}

function nextStrategyFamily(
  shape: ProblemShape | null,
  classification: ProgressClassification,
): StrategyFamily {
  if (classification === "completed") {
    return "wrap_up";
  }
  switch (shape) {
    case "information_gap":
      return "read_context";
    case "implementation_blocked":
      return "try_alternative";
    case "validation_failure":
      return "inspect_failure";
    case "tool_or_permission_limit":
      return "request_capability";
    case "search_drift":
      return "recenter_goal";
    case "parallelizable_subproblems":
      return "split_subproblems";
    default:
      return classification === "advancing" ? "validate_result" : "inspect_failure";
  }
}

function buildVisibilityLabel(
  strategyFamily: StrategyFamily,
  classification: ProgressClassification,
): string {
  if (classification === "completed") {
    return "wrapping up";
  }
  if (strategyFamily === "read_context") {
    return "checking current state";
  }
  if (strategyFamily === "inspect_failure") {
    return "inspecting the blocker";
  }
  if (strategyFamily === "try_alternative") {
    return "trying a different path";
  }
  if (strategyFamily === "validate_result") {
    return "validating the result";
  }
  if (strategyFamily === "request_capability") {
    return "checking the boundary";
  }
  if (strategyFamily === "recenter_goal") {
    return "re-centering on the goal";
  }
  if (strategyFamily === "split_subproblems") {
    return "splitting the work";
  }
  return "still progressing";
}

function advanceDirectionNode(
  state: GoalLoopState,
  classification: ProgressClassification,
): number {
  if (state.directionNodes.length === 0) {
    return 0;
  }
  if (classification === "completed") {
    return Math.max(0, state.directionNodes.length - 1);
  }
  if (classification !== "advancing") {
    return state.currentNodeIndex;
  }
  return Math.min(state.currentNodeIndex + 1, Math.max(0, state.directionNodes.length - 1));
}

export function reviewGoalProgress(args: {
  state: GoalLoopState;
  attempt: EmbeddedRunAttemptResult;
  recentToolCalls: ToolCallRecord[];
  priorResultHashes: Set<string>;
  recentLoopEvents: ToolLoopEventRecord[];
  promptErrorText?: string;
}): GoalProgressReview {
  const { signal, errorFingerprint } = deriveProgressSignal(args);
  const classification = classifyProgress(signal);
  const problemShape = classifyProblemShape({
    attempt: args.attempt,
    recentToolCalls: args.recentToolCalls,
    signal,
    promptErrorText: args.promptErrorText,
  });
  let strategyFamily = nextStrategyFamily(problemShape, classification);
  let stuckCounter =
    classification === "completed" || classification === "advancing"
      ? 0
      : args.state.stuckCounter + (classification === "stalled" ? 2 : 1);
  let shiftCount = args.state.shiftCount;
  let loopGuardLevel: GoalLoopGuardLevel = "normal";

  if (signal.toolLoopEvent === "critical" || stuckCounter >= 4) {
    loopGuardLevel = "force_shift";
    if (strategyFamily === args.state.strategyFamily) {
      strategyFamily = strategyFamily === "try_alternative" ? "inspect_failure" : "try_alternative";
    }
    shiftCount += 1;
  } else if (signal.toolLoopEvent === "warning" || stuckCounter >= 2) {
    loopGuardLevel = "warning";
  }

  if (shiftCount >= 3 || stuckCounter >= 6) {
    loopGuardLevel = "stop";
  }

  const nextState: GoalLoopState = {
    ...args.state,
    attemptCount: args.state.attemptCount + 1,
    currentNodeIndex: advanceDirectionNode(args.state, classification),
    strategyFamily,
    problemShape: problemShape ?? args.state.problemShape,
    progressWindow: [...args.state.progressWindow, signal].slice(-6),
    stuckCounter,
    shiftCount,
    loopGuardLevel,
    convergeReason:
      classification === "completed"
        ? "sufficient_completion"
        : loopGuardLevel === "stop"
          ? "no_progress_no_new_strategy"
          : args.state.convergeReason,
    lastErrorFingerprint: errorFingerprint,
  };
  return {
    signal,
    classification,
    problemShape,
    strategyFamily,
    steeringContext: buildGoalSteeringContext(nextState),
    visibility: buildVisibilityLabel(strategyFamily, classification),
    state: nextState,
  };
}

export function isGoalLoopSuccessful(
  state: GoalLoopState,
  attempt: EmbeddedRunAttemptResult,
): boolean {
  if (state.convergeReason === "sufficient_completion") {
    return true;
  }
  return (
    state.loopGuardLevel !== "stop" &&
    !attempt.clientToolCall &&
    !attempt.promptError &&
    attempt.lastAssistant?.stopReason !== "error" &&
    attempt.assistantTexts.some((text) => text.trim().length > 0)
  );
}

export function buildStrategyMemorySummary(state: GoalLoopState): string {
  const node =
    state.directionNodes[state.currentNodeIndex] ?? state.directionNodes.at(-1) ?? "wrap up";
  const segments = [
    `Objective: ${state.goalFrame.objective}`,
    `Problem shape: ${state.problemShape ?? "unknown"}`,
    `Strategy family: ${state.strategyFamily}`,
    `Current node: ${node}`,
  ];
  return segments.join(" | ");
}

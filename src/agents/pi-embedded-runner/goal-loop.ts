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
  | "synthesize_tool"
  | "delegated_subagent"
  | "recenter_goal"
  | "split_subproblems"
  | "wrap_up";

export type StrategyTrack =
  | "local_execution"
  | "discovery_or_validation"
  | "delegated_subagent"
  | "escalate_or_stop";

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
  strategyTrack: StrategyTrack;
  problemShape: ProblemShape | null;
  progressWindow: ProgressSignal[];
  stuckCounter: number;
  shiftCount: number;
  delegatedShiftCount: number;
  loopGuardLevel: GoalLoopGuardLevel;
  convergeReason: string | null;
  priorStrategyHints: StrategyHint[];
  attemptCount: number;
  lastErrorFingerprint: string | null;
  attemptedDelegationKeys: string[];
  activeDelegation: GoalDelegationPlan | null;
};

export type GoalProgressReview = {
  signal: ProgressSignal;
  classification: ProgressClassification;
  problemShape: ProblemShape | null;
  strategyFamily: StrategyFamily;
  strategyTrack: StrategyTrack;
  delegation: GoalDelegationPlan | null;
  steeringContext: string | null;
  visibility: string;
  state: GoalLoopState;
};

export type GoalDelegationPlan = {
  roles: string[];
  taskGroup: string;
  waitForAll: true;
  announceMode: "aggregate";
  rationale: string;
  key: string;
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
const BINARY_FILE_RE = /detectedMimeType|binary content|cannot display binary/i;
const MAX_DELEGATED_SHIFTS = 2;
const MAX_DELEGATED_FANOUT = 3;

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

function resolveInitialStrategyFamily(_goalType: GoalFrame["goalType"]): StrategyFamily {
  return "read_context";
}

function resolveStrategyTrack(strategyFamily: StrategyFamily): StrategyTrack {
  if (strategyFamily === "delegated_subagent") {
    return "delegated_subagent";
  }
  if (
    strategyFamily === "read_context" ||
    strategyFamily === "validate_result" ||
    strategyFamily === "inspect_failure"
  ) {
    return "discovery_or_validation";
  }
  if (
    strategyFamily === "request_capability" ||
    strategyFamily === "synthesize_tool" ||
    strategyFamily === "wrap_up"
  ) {
    return "escalate_or_stop";
  }
  return "local_execution";
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
    strategyTrack: resolveStrategyTrack(resolveInitialStrategyFamily(goalFrame.goalType)),
    problemShape: null,
    progressWindow: [],
    stuckCounter: 0,
    shiftCount: 0,
    delegatedShiftCount: 0,
    loopGuardLevel: "normal",
    convergeReason: null,
    priorStrategyHints: args.priorStrategyHints ?? [],
    attemptCount: 0,
    lastErrorFingerprint: null,
    attemptedDelegationKeys: [],
    activeDelegation: null,
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
  state: Pick<
    GoalLoopState,
    | "goalFrame"
    | "strategyFamily"
    | "strategyTrack"
    | "problemShape"
    | "loopGuardLevel"
    | "activeDelegation"
  >,
): string | null {
  if (state.strategyFamily === "delegated_subagent" && state.activeDelegation) {
    return (
      `Local attempts are stalling. Use task_dispatch once with waitForAll=true and announceMode="aggregate". ` +
      `Dispatch roles: ${state.activeDelegation.roles.join(", ")}. ` +
      `Task group: ${state.activeDelegation.taskGroup}. ` +
      state.activeDelegation.rationale
    );
  }
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
  if (state.strategyFamily === "synthesize_tool") {
    return "No existing tool covers this capability. Write a targeted script (Python or Bash) to solve the immediate problem, test it, then persist it as a managed skill at ~/.marv/skills/ for future reuse.";
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
  // Require tool usage or new evidence beyond just emitting text —
  // otherwise a trivial "Let me think…" end_turn is a false completion.
  if (
    signal.cleanCompletion &&
    signal.deliveredOutput &&
    (signal.toolBreadth >= 1 || signal.newEvidence || signal.attemptIndex >= 2)
  ) {
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
    // Only flag low-breadth as stalled after the first attempt — on attempt 1,
    // single-tool usage (e.g. reading one file) is legitimate exploration.
    (!signal.errorStateChanged && signal.toolBreadth <= 1 && signal.attemptIndex >= 2)
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
  if (BINARY_FILE_RE.test(errorText)) {
    return "tool_or_permission_limit";
  }
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
  attemptIndex = 0,
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
      return attemptIndex >= 2 ? "synthesize_tool" : "request_capability";
    case "search_drift":
      return "recenter_goal";
    case "parallelizable_subproblems":
      return "split_subproblems";
    default:
      if (classification === "advancing") {
        return "validate_result";
      }
      // When the problem shape is unknown, prefer gathering context on
      // early attempts rather than inspecting a failure that may not exist.
      return attemptIndex <= 1 ? "read_context" : "recenter_goal";
  }
}

function dedupeRoleList(roles: string[]): string[] {
  return dedupeStrings(roles).slice(0, MAX_DELEGATED_FANOUT);
}

function buildDelegationKey(args: {
  goalType: GoalFrame["goalType"];
  problemShape: ProblemShape | null;
  roles: string[];
}): string {
  return `${args.goalType}:${args.problemShape ?? "unknown"}:${args.roles.join(",")}`;
}

function resolveDelegationPlan(args: {
  state: GoalLoopState;
  classification: ProgressClassification;
  problemShape: ProblemShape | null;
  canDelegate: boolean;
}): GoalDelegationPlan | null {
  if (!args.canDelegate) {
    return null;
  }
  if (args.classification === "completed" || args.classification === "advancing") {
    return null;
  }
  if (args.problemShape === "tool_or_permission_limit") {
    return null;
  }

  let roles: string[] = [];
  let rationale = "";
  if (args.state.goalFrame.goalType === "mutation") {
    switch (args.problemShape) {
      case "implementation_blocked":
        roles =
          args.state.goalFrame.complexity === "complex" ? ["debugger", "coder"] : ["debugger"];
        rationale =
          "Have a fresh worker re-attack the implementation blocker from a clean context.";
        break;
      case "validation_failure":
        roles = ["reviewer", "tester"];
        rationale = "Separate failure inspection from read-only validation before more edits.";
        break;
      case "parallelizable_subproblems":
        roles = ["architect", "coder", "tester"];
        rationale =
          "Split design, implementation, and validation into one coordinated aggregate pass.";
        break;
      case "search_drift":
        roles = ["reviewer"];
        rationale =
          "Use a focused reviewer to cut through drift and identify the most credible next move.";
        break;
      default:
        if (
          args.state.loopGuardLevel === "force_shift" &&
          args.state.goalFrame.complexity === "complex"
        ) {
          roles = ["reviewer"];
          rationale = "Take one independent review pass before giving up on the task.";
        }
        break;
    }
  } else {
    switch (args.problemShape) {
      case "information_gap":
        roles = ["researcher"];
        rationale = "Gather missing evidence in a clean context before continuing locally.";
        break;
      case "search_drift":
        roles = ["researcher", "fact_checker"];
        rationale =
          "Split evidence gathering from verification to recover from broad or noisy search.";
        break;
      case "parallelizable_subproblems":
        roles = ["researcher", "fact_checker", "analyst"];
        rationale =
          "Parallelize evidence gathering, verification, and synthesis into one aggregate pass.";
        break;
      default:
        if (args.state.loopGuardLevel === "force_shift") {
          roles = ["analyst"];
          rationale = "Use one fresh analytical pass before stopping for stagnation.";
        }
        break;
    }
  }

  const cappedRoles = dedupeRoleList(roles);
  if (cappedRoles.length === 0) {
    return null;
  }
  const key = buildDelegationKey({
    goalType: args.state.goalFrame.goalType,
    problemShape: args.problemShape,
    roles: cappedRoles,
  });
  if (args.state.attemptedDelegationKeys.includes(key)) {
    return null;
  }
  return {
    roles: cappedRoles,
    taskGroup: `goal-loop:${args.problemShape ?? "recovery"}`,
    waitForAll: true,
    announceMode: "aggregate",
    rationale,
    key,
  };
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
  if (strategyFamily === "synthesize_tool") {
    return "building the missing tool";
  }
  if (strategyFamily === "delegated_subagent") {
    return "dispatching a focused recovery team";
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

/** Pick a diverse strategy when force-shifting, considering problem context. */
function pickForceShiftStrategy(
  current: StrategyFamily,
  problemShape: ProblemShape | null,
  goalFrame: GoalFrame,
): StrategyFamily {
  // Build a prioritized rotation order based on context.
  const candidates: StrategyFamily[] =
    goalFrame.goalType === "mutation"
      ? [
          "recenter_goal",
          "validate_result",
          "try_alternative",
          "inspect_failure",
          "split_subproblems",
        ]
      : [
          "recenter_goal",
          "read_context",
          "inspect_failure",
          "try_alternative",
          "split_subproblems",
        ];

  // Prefer shape-specific strategies when available.
  if (problemShape === "search_drift") {
    candidates.unshift("recenter_goal");
  } else if (problemShape === "validation_failure") {
    candidates.unshift("validate_result");
  } else if (problemShape === "parallelizable_subproblems") {
    candidates.unshift("split_subproblems");
  }

  // Pick the first candidate that differs from the current strategy.
  for (const candidate of candidates) {
    if (candidate !== current) {
      return candidate;
    }
  }
  return current === "try_alternative" ? "inspect_failure" : "try_alternative";
}

export function reviewGoalProgress(args: {
  state: GoalLoopState;
  attempt: EmbeddedRunAttemptResult;
  recentToolCalls: ToolCallRecord[];
  priorResultHashes: Set<string>;
  recentLoopEvents: ToolLoopEventRecord[];
  promptErrorText?: string;
  canDelegate?: boolean;
}): GoalProgressReview {
  const { signal, errorFingerprint } = deriveProgressSignal(args);
  let classification = classifyProgress(signal);
  // For mutation goals, downgrade "completed" to "advancing" when no
  // write-like tool was used — the model likely just narrated without acting.
  if (
    classification === "completed" &&
    args.state.goalFrame.goalType === "mutation" &&
    !hasRecentToolMatching(args.recentToolCalls, WRITE_LIKE_TOOL_RE)
  ) {
    classification = "advancing";
  }
  const problemShape = classifyProblemShape({
    attempt: args.attempt,
    recentToolCalls: args.recentToolCalls,
    signal,
    promptErrorText: args.promptErrorText,
  });
  let strategyFamily = nextStrategyFamily(problemShape, classification, args.state.attemptCount);
  let strategyTrack = resolveStrategyTrack(strategyFamily);
  let stuckIncrement =
    classification === "completed" || classification === "advancing"
      ? 0
      : classification === "stalled"
        ? 2
        : 1;
  // Trend escalation: if the last 3+ progressWindow entries were all
  // non-advancing (ambiguous/stalled), accelerate the stuck counter so
  // prolonged ambiguous drifts don't burn tokens for 6+ rounds.
  if (stuckIncrement > 0 && args.state.progressWindow.length >= 2) {
    const recentNonAdvancing = args.state.progressWindow
      .slice(-3)
      .every((prior) => !prior.newEvidence || prior.resultDiversity < 0.3);
    if (recentNonAdvancing) {
      stuckIncrement += 1;
    }
  }
  let stuckCounter = stuckIncrement === 0 ? 0 : args.state.stuckCounter + stuckIncrement;
  let shiftCount = args.state.shiftCount;
  let delegatedShiftCount = args.state.delegatedShiftCount;
  let loopGuardLevel: GoalLoopGuardLevel = "normal";
  const attemptedDelegationKeys = [...args.state.attemptedDelegationKeys];
  if (
    args.state.strategyFamily === "delegated_subagent" &&
    args.state.activeDelegation &&
    classification !== "completed" &&
    classification !== "advancing" &&
    !attemptedDelegationKeys.includes(args.state.activeDelegation.key)
  ) {
    attemptedDelegationKeys.push(args.state.activeDelegation.key);
  }
  let delegation: GoalDelegationPlan | null = null;
  const canDelegate = args.canDelegate !== false;

  if (signal.toolLoopEvent === "critical" || stuckCounter >= 4) {
    loopGuardLevel = "force_shift";
    if (strategyFamily === args.state.strategyFamily) {
      // Rotate through a broader set of strategies based on problem context,
      // instead of only flipping between try_alternative and inspect_failure.
      strategyFamily = pickForceShiftStrategy(
        args.state.strategyFamily,
        problemShape,
        args.state.goalFrame,
      );
    }
    shiftCount += 1;
  } else if (signal.toolLoopEvent === "warning" || stuckCounter >= 2) {
    loopGuardLevel = "warning";
  }

  const delegatedRecovery =
    delegatedShiftCount < MAX_DELEGATED_SHIFTS
      ? resolveDelegationPlan({
          state: {
            ...args.state,
            attemptedDelegationKeys,
          },
          classification,
          problemShape,
          canDelegate,
        })
      : null;

  if (
    delegatedRecovery &&
    (loopGuardLevel === "force_shift" || stuckCounter >= 2 || classification === "stalled")
  ) {
    strategyFamily = "delegated_subagent";
    strategyTrack = "delegated_subagent";
    delegation = delegatedRecovery;
    if (
      args.state.strategyFamily !== "delegated_subagent" ||
      args.state.activeDelegation?.key !== delegatedRecovery.key
    ) {
      shiftCount += 1;
      delegatedShiftCount += 1;
    }
  } else {
    strategyTrack = resolveStrategyTrack(strategyFamily);
  }

  if (shiftCount >= 3 || stuckCounter >= 6) {
    loopGuardLevel = "stop";
  }

  // Last-resort rescue: if we're about to stop and a delegation plan is
  // available that hasn't been tried yet, use it as a final recovery attempt.
  // This rescue can fire at most once per unique delegation key (guarded by
  // attemptedDelegationKeys dedup) and is capped by MAX_DELEGATED_SHIFTS.
  if (loopGuardLevel === "stop" && delegatedRecovery) {
    if (delegation === null) {
      strategyFamily = "delegated_subagent";
      strategyTrack = "delegated_subagent";
      delegation = delegatedRecovery;
      shiftCount += 1;
      delegatedShiftCount += 1;
    }
    // Downgrade stop → force_shift to allow the delegation attempt to run.
    loopGuardLevel = "force_shift";
  }

  const nextState: GoalLoopState = {
    ...args.state,
    attemptCount: args.state.attemptCount + 1,
    currentNodeIndex: advanceDirectionNode(args.state, classification),
    strategyFamily,
    strategyTrack,
    problemShape: problemShape ?? args.state.problemShape,
    progressWindow: [...args.state.progressWindow, signal].slice(-6),
    stuckCounter,
    shiftCount,
    delegatedShiftCount,
    loopGuardLevel,
    convergeReason:
      classification === "completed"
        ? "sufficient_completion"
        : loopGuardLevel === "stop"
          ? "no_progress_no_new_strategy"
          : args.state.convergeReason,
    lastErrorFingerprint: errorFingerprint,
    attemptedDelegationKeys: attemptedDelegationKeys.slice(-6),
    activeDelegation: delegation,
  };
  return {
    signal,
    classification,
    problemShape,
    strategyFamily,
    strategyTrack,
    delegation,
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

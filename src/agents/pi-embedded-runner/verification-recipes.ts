/**
 * O3: Domain-specific verification recipes.
 * Replaces the generic "validate your work" guidance with concrete checklists
 * matched to the type of work the agent has been doing.
 */

import type { ToolCallRecord } from "../../logging/diagnostic-session-state.js";
import type { GoalFrame } from "./goal-loop-types.js";

export type VerificationDomain = "code" | "messaging" | "research" | "config" | "general";

const CODE_TOOL_RE = /^(write|edit|apply_patch|exec|process)$/;
const MESSAGING_TOOL_RE = /^(message|sessions_send)$/;
const RESEARCH_TOOL_RE = /^(web_search|web_fetch|memory_search)$/;
const CONFIG_TOOL_RE = /^gateway$/;

/**
 * Determine the verification domain from recent tool usage patterns.
 */
export function resolveVerificationDomain(params: {
  goalType: GoalFrame["goalType"];
  recentToolCalls: ToolCallRecord[];
  objective: string;
}): VerificationDomain {
  const toolNames = params.recentToolCalls.map((tc) => tc.toolName);
  const counts = {
    code: toolNames.filter((n) => CODE_TOOL_RE.test(n)).length,
    messaging: toolNames.filter((n) => MESSAGING_TOOL_RE.test(n)).length,
    research: toolNames.filter((n) => RESEARCH_TOOL_RE.test(n)).length,
    config: toolNames.filter((n) => CONFIG_TOOL_RE.test(n)).length,
  };
  // Pick the domain with the most recent tool usage
  const max = Math.max(counts.code, counts.messaging, counts.research, counts.config);
  if (max === 0) {
    return "general";
  }
  if (counts.code === max) {
    return "code";
  }
  if (counts.messaging === max) {
    return "messaging";
  }
  if (counts.research === max) {
    return "research";
  }
  if (counts.config === max) {
    return "config";
  }
  return "general";
}

const RECIPES: Record<VerificationDomain, string> = {
  code: [
    "Verification checklist:",
    "1. Run existing tests or linters if available.",
    "2. Check types or build output for errors.",
    "3. Read modified files to confirm the change is correct.",
    "4. Compare behavior with the original state.",
  ].join("\n"),
  messaging: [
    "Verification checklist:",
    "1. Confirm delivery status of sent messages.",
    "2. Verify formatting renders correctly on the target channel.",
    "3. Check recipient and thread were correct.",
    "4. Verify no duplicate or unintended sends.",
  ].join("\n"),
  research: [
    "Verification checklist:",
    "1. Cross-reference key claims across multiple sources.",
    "2. Verify information recency.",
    "3. Check completeness against the original question.",
    "4. Note remaining uncertainty.",
  ].join("\n"),
  config: [
    "Verification checklist:",
    "1. Validate config syntax or schema.",
    "2. Test affected functionality with the new config.",
    "3. Confirm no regression in existing behavior.",
    "4. Verify changes are reversible.",
  ].join("\n"),
  general: [
    "Verification checklist:",
    "1. Re-read the original request.",
    "2. Verify output matches what was asked.",
    "3. Check for edge cases or omissions.",
    "4. Summarize what was done and remaining limitations.",
  ].join("\n"),
};

/**
 * Build a verification checklist for the given domain.
 * When scaffoldLevel is "lean", returns a brief single-line hint instead.
 */
export function buildVerificationChecklist(
  domain: VerificationDomain,
  scaffoldLevel?: "full" | "standard" | "lean",
): string {
  if (scaffoldLevel === "lean") {
    return "Prefer validation over more edits. Run the narrowest useful check.";
  }
  return RECIPES[domain] ?? RECIPES.general;
}

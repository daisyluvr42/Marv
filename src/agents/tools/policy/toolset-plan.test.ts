import { describe, expect, it } from "vitest";
import {
  TOOLSET_INTENT_RESOLUTION_PRIORITY,
  TOOLSET_INTENTS,
  applyToolsetPlanToToolNames,
  createToolsetPlan,
  expandSuppressedToolsForIntent,
  detectExplicitToolMentions,
  resolveToolsetSelectionMode,
  shouldForceMixedIntentFromInstruction,
} from "./toolset-plan.js";

describe("toolset plan phase-0 contract", () => {
  it("keeps the revised priority order stable", () => {
    expect(TOOLSET_INTENT_RESOLUTION_PRIORITY).toEqual([
      "explicit_instruction",
      "command",
      "channel",
      "tool_profile",
      "fallback",
    ]);
    expect(TOOLSET_INTENTS).toEqual(["coding", "research", "messaging", "operator", "mixed"]);
  });

  it("expands suppressed tools deterministically for coding intent", () => {
    expect(expandSuppressedToolsForIntent("coding")).toEqual([
      "gateway",
      "cron",
      "browser",
      "canvas",
      "message",
      "tts",
    ]);
  });

  it("leaves operator and mixed intents unsuppressed", () => {
    expect(expandSuppressedToolsForIntent("operator")).toEqual([]);
    expect(expandSuppressedToolsForIntent("mixed")).toEqual([]);
  });

  it("detects explicit tool mentions via aliases", () => {
    expect(
      detectExplicitToolMentions("Use the browser, then apply patch and run a shell command."),
    ).toEqual(["apply_patch", "browser", "exec"]);
  });

  it("forces mixed intent when the instruction explicitly asks for a suppressed tool", () => {
    expect(
      shouldForceMixedIntentFromInstruction({
        instruction: "Please use the browser for this one.",
        intent: "research",
      }),
    ).toBe(true);
    expect(
      shouldForceMixedIntentFromInstruction({
        instruction: "Just analyze this code path.",
        intent: "research",
      }),
    ).toBe(false);
  });

  it("defaults enabled selection config to observe mode", () => {
    expect(
      resolveToolsetSelectionMode({
        cfg: {
          agents: {
            defaults: {
              tools: {
                selection: {
                  enabled: true,
                },
              },
            },
          },
        },
        agentId: "main",
      }),
    ).toBe("observe");
  });

  it("resolves coding intent from task context and filters suppressed tools in enforce mode", () => {
    const plan = createToolsetPlan({
      mode: "enforce",
      taskId: "task_123",
      counts: {
        toolNames: ["read", "write", "exec", "message", "session_status"],
      },
    });

    expect(plan.intent).toBe("coding");
    expect(plan.suppressedTools).toEqual(["message"]);
    expect(plan.effectiveToolCount).toBe(4);
    expect(applyToolsetPlanToToolNames(["read", "write", "exec", "message"], plan)).toEqual([
      "read",
      "write",
      "exec",
    ]);
  });

  it("resolves mixed intent when instruction and channel signals conflict at the top priority", () => {
    const plan = createToolsetPlan({
      mode: "observe",
      instruction: "Please investigate this and send a message back.",
      messageProvider: "telegram",
      counts: {
        toolNames: ["web_search", "message"],
      },
    });

    expect(plan.intent).toBe("mixed");
    expect(plan.reasons).toContain("conflicting explicit_instruction signals");
    expect(plan.effectiveToolCount).toBe(2);
  });
});

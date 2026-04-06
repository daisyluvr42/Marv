import { describe, expect, it } from "vitest";
import { buildAnnounceReplyInstruction } from "./subagent-announce.js";

describe("buildAnnounceReplyInstruction", () => {
  const NO_HALLUCINATE = "Do NOT promise";
  const TOOL_CALL_GUARD = "tool call in this same turn";

  it("cron job delivery forbids hallucinated follow-up actions", () => {
    const instruction = buildAnnounceReplyInstruction({
      remainingActiveSubagentRuns: 0,
      requesterIsSubagent: false,
      announceType: "cron job",
    });
    expect(instruction).toContain(NO_HALLUCINATE);
    expect(instruction).toContain(TOOL_CALL_GUARD);
  });

  it("expectsCompletionMessage variant also forbids hallucinated actions", () => {
    const instruction = buildAnnounceReplyInstruction({
      remainingActiveSubagentRuns: 0,
      requesterIsSubagent: false,
      announceType: "cron job",
      expectsCompletionMessage: true,
    });
    expect(instruction).toContain(NO_HALLUCINATE);
    expect(instruction).toContain(TOOL_CALL_GUARD);
  });

  it("subagent task delivery also forbids hallucinated actions", () => {
    const instruction = buildAnnounceReplyInstruction({
      remainingActiveSubagentRuns: 0,
      requesterIsSubagent: false,
      announceType: "subagent task",
    });
    expect(instruction).toContain(NO_HALLUCINATE);
    expect(instruction).toContain(TOOL_CALL_GUARD);
  });

  it("orchestration update (requesterIsSubagent) does not need the guard", () => {
    const instruction = buildAnnounceReplyInstruction({
      remainingActiveSubagentRuns: 0,
      requesterIsSubagent: true,
      announceType: "subagent task",
    });
    // Internal orchestration updates don't face users, so no hallucination guard needed
    expect(instruction).not.toContain(NO_HALLUCINATE);
  });
});

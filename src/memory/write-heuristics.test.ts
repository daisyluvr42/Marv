import { describe, expect, it } from "vitest";
import { evaluateMemoryWriteHeuristics } from "./write-heuristics.js";

describe("evaluateMemoryWriteHeuristics", () => {
  it("accepts explicit remember directives and strips the prefix", () => {
    const result = evaluateMemoryWriteHeuristics({
      content: "Please remember that I prefer concise Chinese replies.",
      kind: "note",
    });
    expect(result).toEqual({
      shouldWrite: true,
      classification: "explicit_memory",
      normalizedContent: "I prefer concise Chinese replies.",
    });
  });

  it("accepts durable preferences without explicit remember wording", () => {
    const result = evaluateMemoryWriteHeuristics({
      content: "I prefer concise Chinese replies with concrete dates.",
      kind: "preference",
    });
    expect(result).toMatchObject({
      shouldWrite: true,
      classification: "durable_preference",
    });
  });

  it("rejects question-shaped writes", () => {
    const result = evaluateMemoryWriteHeuristics({
      content: "Can you check whether the deploy is healthy right now?",
      kind: "note",
    });
    expect(result).toEqual({
      shouldWrite: false,
      classification: "reject_question",
      normalizedContent: "Can you check whether the deploy is healthy right now?",
    });
  });

  it("rejects transient task chatter", () => {
    const result = evaluateMemoryWriteHeuristics({
      content: "We are debugging the flaky deploy right now.",
      kind: "note",
    });
    expect(result).toEqual({
      shouldWrite: false,
      classification: "reject_transient",
      normalizedContent: "We are debugging the flaky deploy right now.",
    });
  });
});

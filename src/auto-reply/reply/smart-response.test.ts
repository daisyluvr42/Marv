import { describe, expect, it } from "vitest";
import {
  classifyWithHeuristics,
  buildClassifierPrompt,
  parseClassifierResponse,
  evaluateSmartResponse,
} from "./smart-response.js";

describe("classifyWithHeuristics", () => {
  const base = { messageBody: "hello", wasMentioned: false, isReplyToBot: false };

  it("returns respond when mentioned", () => {
    expect(classifyWithHeuristics({ ...base, wasMentioned: true })).toBe("respond");
  });

  it("returns respond when reply to bot", () => {
    expect(classifyWithHeuristics({ ...base, isReplyToBot: true })).toBe("respond");
  });

  it("returns skip for empty message", () => {
    expect(classifyWithHeuristics({ ...base, messageBody: "  " })).toBe("skip");
  });

  it("returns skip for noise messages", () => {
    for (const noise of ["lol", "haha", "ok", "👍", "😂", "nice", "cool", "thx"]) {
      expect(classifyWithHeuristics({ ...base, messageBody: noise })).toBe("skip");
    }
  });

  it("returns respond when bot name is mentioned in text", () => {
    expect(
      classifyWithHeuristics({ ...base, messageBody: "hey Marv can you help?", botName: "Marv" }),
    ).toBe("respond");
  });

  it("returns uncertain for questions with you", () => {
    expect(classifyWithHeuristics({ ...base, messageBody: "can you help me?" })).toBe("uncertain");
  });

  it("returns uncertain for general questions", () => {
    expect(classifyWithHeuristics({ ...base, messageBody: "what time is it?" })).toBe("uncertain");
  });

  it("returns skip for rapid human-to-human exchange", () => {
    const recentHistory = [
      { sender: "Alice", body: "hey" },
      { sender: "Bob", body: "sup" },
      { sender: "Alice", body: "let's go" },
    ];
    expect(classifyWithHeuristics({ ...base, messageBody: "sounds good", recentHistory })).toBe(
      "skip",
    );
  });

  it("returns respond when custom pattern matches", () => {
    expect(
      classifyWithHeuristics({
        ...base,
        messageBody: "please deploy the app",
        alwaysRespondPatterns: ["deploy"],
      }),
    ).toBe("respond");
  });

  it("ignores invalid regex in custom patterns", () => {
    expect(
      classifyWithHeuristics({
        ...base,
        messageBody: "test",
        alwaysRespondPatterns: ["[invalid"],
      }),
    ).toBe("uncertain");
  });
});

describe("buildClassifierPrompt", () => {
  it("includes bot name and message", () => {
    const prompt = buildClassifierPrompt({
      botName: "Marv",
      senderName: "Alice",
      messageBody: "what's for lunch?",
    });
    expect(prompt).toContain("Marv");
    expect(prompt).toContain("Alice");
    expect(prompt).toContain("what's for lunch?");
    expect(prompt).toContain("RESPOND or SKIP");
  });

  it("includes recent context when provided", () => {
    const prompt = buildClassifierPrompt({
      botName: "Marv",
      senderName: "Bob",
      messageBody: "help?",
      recentContext: [{ sender: "Alice", body: "let's discuss the project" }],
    });
    expect(prompt).toContain("Alice: let's discuss the project");
  });
});

describe("parseClassifierResponse", () => {
  it("parses RESPOND", () => {
    expect(parseClassifierResponse("RESPOND\nBecause...")).toBe("respond");
  });

  it("parses SKIP", () => {
    expect(parseClassifierResponse("SKIP\nNot relevant")).toBe("skip");
  });

  it("defaults to skip on unclear response", () => {
    expect(parseClassifierResponse("I think maybe")).toBe("skip");
  });

  it("handles lowercase", () => {
    expect(parseClassifierResponse("respond")).toBe("respond");
  });
});

describe("evaluateSmartResponse", () => {
  const base = {
    messageBody: "hello",
    wasMentioned: false,
    isReplyToBot: false,
    sessionKey: "test-session",
  };

  it("returns skip for noise messages", () => {
    expect(evaluateSmartResponse({ ...base, messageBody: "lol" })).toBe("skip");
  });

  it("returns respond for mentioned messages", () => {
    expect(evaluateSmartResponse({ ...base, wasMentioned: true })).toBe("respond");
  });

  it("returns skip for uncertain without classifier decision", () => {
    expect(evaluateSmartResponse({ ...base, messageBody: "what time?" })).toBe("skip");
  });

  it("returns respond for uncertain with classifier decision", () => {
    expect(
      evaluateSmartResponse({
        ...base,
        messageBody: "what time?",
        classifierDecision: "respond",
      }),
    ).toBe("respond");
  });

  it("respects rate limiter", () => {
    const key = `rate-limit-test-${Date.now()}`;
    const config = { maxAutoResponsesPerMinute: 2 };
    const params = { ...base, sessionKey: key, config, botName: "Bot" };

    // Bot name in text triggers "respond" heuristic (non-mention).
    const withBotName = { ...params, messageBody: "hey Bot what's up" };
    expect(evaluateSmartResponse(withBotName)).toBe("respond");
    expect(evaluateSmartResponse(withBotName)).toBe("respond");
    // Third should be rate limited.
    expect(evaluateSmartResponse(withBotName)).toBe("skip");
  });

  it("mentions bypass rate limiter", () => {
    const key = `rate-limit-mention-test-${Date.now()}`;
    const config = { maxAutoResponsesPerMinute: 1 };
    const mentionParams = { ...base, sessionKey: key, config, wasMentioned: true };

    expect(evaluateSmartResponse(mentionParams)).toBe("respond");
    // Even though rate limit is 1, mentions bypass it.
    expect(evaluateSmartResponse(mentionParams)).toBe("respond");
  });
});

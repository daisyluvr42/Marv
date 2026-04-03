import { describe, expect, it } from "vitest";
import { sanitizeUserFacingText } from "../../agents/pi-embedded-helpers/errors.js";

/**
 * Regression tests: echoed inbound metadata blocks must be stripped
 * from assistant replies before delivery to users.
 */
describe("sanitizeUserFacingText strips echoed inbound metadata", () => {
  it("strips a single Conversation info metadata block", () => {
    const text = [
      "Conversation info (untrusted metadata):",
      "```json",
      '{"message_id":"abc123","sender":"Alice"}',
      "```",
      "",
      "Sure, I can help with that!",
    ].join("\n");
    expect(sanitizeUserFacingText(text)).toBe("Sure, I can help with that!");
  });

  it("strips multiple consecutive metadata blocks", () => {
    const text = [
      "Conversation info (untrusted metadata):",
      "```json",
      '{"message_id":"abc123"}',
      "```",
      "",
      "Sender (untrusted metadata):",
      "```json",
      '{"label":"Alice","name":"Alice Smith"}',
      "```",
      "",
      "Here is my response.",
    ].join("\n");
    expect(sanitizeUserFacingText(text)).toBe("Here is my response.");
  });

  it("strips Thread starter metadata block", () => {
    const text = [
      "Thread starter (untrusted, for context):",
      "```json",
      '{"body":"original message"}',
      "```",
      "",
      "Got it, replying now.",
    ].join("\n");
    expect(sanitizeUserFacingText(text)).toBe("Got it, replying now.");
  });

  it("strips Replied message metadata block", () => {
    const text = [
      "Replied message (untrusted, for context):",
      "```json",
      '{"sender_label":"Bob","is_quote":true,"body":"quoted text"}',
      "```",
      "",
      "Thanks for the context.",
    ].join("\n");
    expect(sanitizeUserFacingText(text)).toBe("Thanks for the context.");
  });

  it("strips Forwarded message context block", () => {
    const text = [
      "Forwarded message context (untrusted metadata):",
      "```json",
      '{"from":"Charlie"}',
      "```",
      "",
      "I see the forwarded message.",
    ].join("\n");
    expect(sanitizeUserFacingText(text)).toBe("I see the forwarded message.");
  });

  it("strips Chat history metadata block", () => {
    const text = [
      "Chat history since last reply (untrusted, for context):",
      "```json",
      '[{"sender":"Alice","body":"hello"}]',
      "```",
      "",
      "Catching up on the history.",
    ].join("\n");
    expect(sanitizeUserFacingText(text)).toBe("Catching up on the history.");
  });

  it("leaves clean text unchanged (fast path)", () => {
    const text = "This is a normal reply with no metadata.";
    expect(sanitizeUserFacingText(text)).toBe(text);
  });

  it("handles metadata blocks combined with <final> tags", () => {
    const text = [
      "<final>",
      "Conversation info (untrusted metadata):",
      "```json",
      '{"conversation_label":"test"}',
      "```",
      "",
      "Hello!</final>",
    ].join("\n");
    expect(sanitizeUserFacingText(text)).toBe("Hello!");
  });

  it("strips metadata when it appears mid-reply", () => {
    const text = [
      "Here is part one.",
      "",
      "Conversation info (untrusted metadata):",
      "```json",
      '{"message_id":"leak"}',
      "```",
      "",
      "And here is part two.",
    ].join("\n");
    const result = sanitizeUserFacingText(text);
    expect(result).toContain("Here is part one.");
    expect(result).toContain("And here is part two.");
    expect(result).not.toContain("untrusted metadata");
    expect(result).not.toContain("message_id");
  });

  it("returns empty string when reply is only metadata", () => {
    const text = [
      "Conversation info (untrusted metadata):",
      "```json",
      '{"message_id":"abc123"}',
      "```",
    ].join("\n");
    expect(sanitizeUserFacingText(text)).toBe("");
  });
});

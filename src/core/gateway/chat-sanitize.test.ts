import { describe, expect, test } from "vitest";
import { stripEnvelopeFromMessage } from "./chat-sanitize.js";

describe("stripEnvelopeFromMessage", () => {
  test("removes message_id hint lines from user messages", () => {
    const input = {
      role: "user",
      content: "[WhatsApp 2026-01-24 13:36] yolo\n[message_id: 7b8b]",
    };
    const result = stripEnvelopeFromMessage(input) as { content?: string };
    expect(result.content).toBe("yolo");
  });

  test("removes message_id hint lines from text content arrays", () => {
    const input = {
      role: "user",
      content: [{ type: "text", text: "hi\n[message_id: abc123]" }],
    };
    const result = stripEnvelopeFromMessage(input) as {
      content?: Array<{ type: string; text?: string }>;
    };
    expect(result.content?.[0]?.text).toBe("hi");
  });

  test("does not strip inline message_id text that is part of a line", () => {
    const input = {
      role: "user",
      content: "I typed [message_id: 123] on purpose",
    };
    const result = stripEnvelopeFromMessage(input) as { content?: string };
    expect(result.content).toBe("I typed [message_id: 123] on purpose");
  });

  test("does not strip assistant messages", () => {
    const input = {
      role: "assistant",
      content: "note\n[message_id: 123]",
    };
    const result = stripEnvelopeFromMessage(input) as { content?: string };
    expect(result.content).toBe("note\n[message_id: 123]");
  });

  test("removes inbound metadata blocks from user string content", () => {
    const input = {
      role: "user",
      content: [
        "Conversation info (untrusted metadata):",
        "```json",
        '{ "message_id": "187", "sender": "936144835" }',
        "```",
        "",
        "？",
      ].join("\n"),
    };
    const result = stripEnvelopeFromMessage(input) as { content?: string };
    expect(result.content).toBe("？");
  });

  test("removes inbound metadata blocks from user text content arrays", () => {
    const input = {
      role: "user",
      content: [
        {
          type: "text",
          text: [
            "Conversation info (untrusted metadata):",
            "```json",
            '{ "message_id": "187", "sender": "936144835" }',
            "```",
            "",
            "Hello",
          ].join("\n"),
        },
      ],
    };
    const result = stripEnvelopeFromMessage(input) as {
      content?: Array<{ type: string; text?: string }>;
    };
    expect(result.content?.[0]?.text).toBe("Hello");
  });
});

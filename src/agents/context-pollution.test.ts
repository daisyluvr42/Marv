import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { describe, expect, it } from "vitest";
import type { TaskContextEntry } from "../memory/task-context/types.js";
import {
  filterTaskContextEntriesForReplyPreferences,
  filterTranscriptMessagesForReplyPreferences,
  scanContextPollution,
  taskContextTurnsFromEntries,
  transcriptTurnsFromMessages,
} from "./context-pollution.js";

describe("context pollution", () => {
  it("detects assistant pinyin after an explicit no-pinyin user preference", () => {
    const scan = scanContextPollution([
      {
        id: "u1",
        role: "user",
        text: "你用中文回复的时候不要带拼音",
        order: 1,
        source: "transcript",
      },
      {
        id: "a1",
        role: "assistant",
        text: "好的，我会注意。(Hǎo de, wǒ huì zhùyì.)",
        order: 2,
        source: "transcript",
        removable: true,
      },
    ]);

    expect(scan.preferences.noPinyinChinese).toBe(true);
    expect(scan.violations).toHaveLength(1);
    expect(scan.violations[0]?.reasons).toContain("pinyin");
  });

  it("filters transcript assistant replies that violate the active no-pinyin preference", () => {
    const messages = [
      { role: "user", content: "以后中文不要带拼音" },
      {
        role: "assistant",
        content: [{ type: "text", text: "明白了。(Míngbái le.)" }],
        api: "x",
        provider: "p",
        model: "m",
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "stop",
        timestamp: 1,
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "后面这条应该保留。" }],
        api: "x",
        provider: "p",
        model: "m",
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "stop",
        timestamp: 2,
      },
    ] as unknown as AgentMessage[];

    const filtered = filterTranscriptMessagesForReplyPreferences([...messages]);
    expect(filtered.removedCount).toBe(1);
    expect(filtered.sanitizedCount).toBe(0);
    expect(filtered.messages).toHaveLength(2);
    expect(transcriptTurnsFromMessages(filtered.messages)[1]?.text).toBe("后面这条应该保留。");
  });

  it("strips polluted assistant text but keeps tool calls in mixed assistant turns", () => {
    const messages = [
      { role: "user", content: "以后中文不要带拼音，也不用整段翻译一遍" },
      {
        role: "assistant",
        content: [
          { type: "text", text: "好的，我先查一下。(Hǎo de, wǒ xiān chá yíxià.)" },
          { type: "toolCall", id: "tc1", name: "self_inspecting", arguments: { query: "models" } },
        ],
        api: "x",
        provider: "p",
        model: "m",
        usage: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
        stopReason: "toolUse",
        timestamp: 1,
      },
    ] as unknown as AgentMessage[];

    const filtered = filterTranscriptMessagesForReplyPreferences([...messages]);
    expect(filtered.removedCount).toBe(0);
    expect(filtered.sanitizedCount).toBe(1);
    const assistant = filtered.messages[1] as { content: unknown[] };
    expect(Array.isArray(assistant.content)).toBe(true);
    expect(assistant.content).toHaveLength(1);
    expect(assistant.content[0]).toMatchObject({ type: "toolCall", name: "self_inspecting" });
  });

  it("filters task-context assistant entries that violate the active no-pinyin preference", () => {
    const entries = [
      {
        id: "u1",
        taskId: "agent-main-main",
        sequence: 1,
        role: "user",
        content: "中文不要带拼音",
        contentHash: "u1",
        tokenCount: 1,
        createdAt: 1,
      },
      {
        id: "a1",
        taskId: "agent-main-main",
        sequence: 2,
        role: "assistant",
        content: "收到。(Shōudào.)",
        contentHash: "a1",
        tokenCount: 1,
        createdAt: 2,
      },
    ] as unknown as TaskContextEntry[];

    const filtered = filterTaskContextEntriesForReplyPreferences([...entries]);
    expect(filtered.removedIds).toEqual(["a1"]);
    expect(taskContextTurnsFromEntries(filtered.entries)).toHaveLength(1);
  });
});

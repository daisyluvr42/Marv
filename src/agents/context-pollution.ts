import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type { TaskContextEntry } from "../memory/task-context/types.js";

export type ReplyFormatPreferences = {
  noPinyinChinese: boolean;
  noInlineEnglishChinese: boolean;
};

export type ContextPollutionTurn = {
  id: string;
  role: "user" | "assistant" | "tool" | "system";
  text: string;
  order: number;
  source: "transcript" | "task-context";
  removable?: boolean;
};

export type ContextPollutionViolationReason = "pinyin" | "inline_english";

export type ContextPollutionViolation = {
  id: string;
  role: "assistant";
  order: number;
  source: "transcript" | "task-context";
  reasons: ContextPollutionViolationReason[];
  text: string;
};

export type ContextPollutionScan = {
  preferences: ReplyFormatPreferences;
  violations: ContextPollutionViolation[];
};

const NO_PINYIN_PATTERNS = [
  /不要.{0,8}拼音/u,
  /不需要.{0,8}拼音/u,
  /不用.{0,8}拼音/u,
  /别.{0,8}拼音/u,
  /无需.{0,8}拼音/u,
  /不会.{0,8}拼音/u,
  /\bno pinyin\b/i,
  /\bwithout pinyin\b/i,
  /\bplain chinese(?: characters)? only\b/i,
  /\bdo not .*pinyin\b/i,
  /\bdon't .*pinyin\b/i,
];

const ALLOW_PINYIN_PATTERNS = [
  /带上.{0,8}拼音/u,
  /加上.{0,8}拼音/u,
  /附上.{0,8}拼音/u,
  /给我.{0,8}拼音/u,
  /请用.{0,8}拼音/u,
  /\bwith pinyin\b/i,
  /\binclude pinyin\b/i,
  /\badd pinyin\b/i,
];

const NO_INLINE_ENGLISH_PATTERNS = [
  /只说中文/u,
  /只用中文/u,
  /不要.{0,10}英文翻译/u,
  /不用.{0,10}英文翻译/u,
  /不需要.{0,10}英文翻译/u,
  /\bonly chinese\b/i,
  /\bno english translation\b/i,
  /\bwithout english translation\b/i,
];

const ALLOW_INLINE_ENGLISH_PATTERNS = [
  /附上.{0,8}英文/u,
  /加上.{0,8}英文/u,
  /带上.{0,8}英文/u,
  /\binclude english\b/i,
  /\bwith english translation\b/i,
];

const PINYIN_SYLLABLE_RE =
  /^(?:zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])?(?:iang|iong|uang|ueng|ang|eng|ing|ong|iao|ian|uan|uen|uai|uei|ai|ei|ao|ou|an|en|in|un|er|ia|ie|ua|uo|ui|iu|ve|ue|a|o|e|i|u|v)$/;

function stripDiacritics(value: string): string {
  return value.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

function containsChinese(text: string): boolean {
  return /[\u3400-\u9fff]/u.test(text);
}

function parentheticalSegments(text: string): string[] {
  return [...text.matchAll(/[（(]([^()（）\n]{3,})[)）]/gu)].map((match) => match[1] ?? "");
}

function countLikelyPinyinWords(segment: string): number {
  const words =
    segment
      .match(/[A-Za-z\u00C0-\u024F\u01CD-\u01DCüÜvV]+/gu)
      ?.map((word) => stripDiacritics(word).toLowerCase())
      .filter(Boolean) ?? [];
  return words.filter((word) => PINYIN_SYLLABLE_RE.test(word)).length;
}

function segmentLooksLikePinyin(segment: string): boolean {
  if (/[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜü]/iu.test(segment)) {
    return true;
  }
  const words =
    segment
      .match(/[A-Za-z\u00C0-\u024F\u01CD-\u01DCüÜvV]+/gu)
      ?.map((word) => word.trim())
      .filter(Boolean) ?? [];
  if (words.length < 3) {
    return false;
  }
  const pinyinWords = countLikelyPinyinWords(segment);
  return pinyinWords >= 3 && pinyinWords / words.length >= 0.6;
}

function segmentLooksLikeInlineEnglish(segment: string): boolean {
  if (!/[A-Za-z]/.test(segment)) {
    return false;
  }
  if (segmentLooksLikePinyin(segment)) {
    return false;
  }
  const englishWords =
    segment
      .match(/[A-Za-z]+/g)
      ?.map((word) => word.toLowerCase())
      .filter((word) => word.length > 1) ?? [];
  return englishWords.length >= 4;
}

export function detectReplyFormatPreferencesFromText(
  text: string,
  current: ReplyFormatPreferences,
): ReplyFormatPreferences {
  let next = current;
  if (NO_PINYIN_PATTERNS.some((pattern) => pattern.test(text))) {
    next = { ...next, noPinyinChinese: true };
  } else if (ALLOW_PINYIN_PATTERNS.some((pattern) => pattern.test(text))) {
    next = { ...next, noPinyinChinese: false };
  }

  if (NO_INLINE_ENGLISH_PATTERNS.some((pattern) => pattern.test(text))) {
    next = { ...next, noInlineEnglishChinese: true };
  } else if (ALLOW_INLINE_ENGLISH_PATTERNS.some((pattern) => pattern.test(text))) {
    next = { ...next, noInlineEnglishChinese: false };
  }
  return next;
}

export function detectContextPollutionReasons(
  text: string,
  preferences: ReplyFormatPreferences,
): ContextPollutionViolationReason[] {
  if (!containsChinese(text)) {
    return [];
  }
  const reasons = new Set<ContextPollutionViolationReason>();
  for (const segment of parentheticalSegments(text)) {
    if (preferences.noPinyinChinese && segmentLooksLikePinyin(segment)) {
      reasons.add("pinyin");
    }
    if (preferences.noInlineEnglishChinese && segmentLooksLikeInlineEnglish(segment)) {
      reasons.add("inline_english");
    }
  }
  return [...reasons];
}

export function scanContextPollution(turns: ContextPollutionTurn[]): ContextPollutionScan {
  let preferences: ReplyFormatPreferences = {
    noPinyinChinese: false,
    noInlineEnglishChinese: false,
  };
  const violations: ContextPollutionViolation[] = [];

  for (const turn of turns) {
    if (turn.role === "user") {
      preferences = detectReplyFormatPreferencesFromText(turn.text, preferences);
      continue;
    }
    if (turn.role !== "assistant") {
      continue;
    }
    const reasons = detectContextPollutionReasons(turn.text, preferences);
    if (reasons.length === 0) {
      continue;
    }
    violations.push({
      id: turn.id,
      role: "assistant",
      order: turn.order,
      source: turn.source,
      reasons,
      text: turn.text,
    });
  }

  return { preferences, violations };
}

function assistantMessageText(message: Extract<AgentMessage, { role: "assistant" }>): string {
  return message.content
    .filter(
      (block): block is Extract<(typeof message.content)[number], { type: "text" }> =>
        block?.type === "text" && typeof block.text === "string",
    )
    .map((block) => block.text.trim())
    .filter(Boolean)
    .join("\n\n");
}

function stripPollutingAssistantTextBlocks(
  message: Extract<AgentMessage, { role: "assistant" }>,
  preferences: ReplyFormatPreferences,
): {
  message: Extract<AgentMessage, { role: "assistant" }>;
  stripped: boolean;
} {
  const nextContent = message.content.filter((block) => {
    if (block?.type !== "text" || typeof block.text !== "string") {
      return true;
    }
    return detectContextPollutionReasons(block.text, preferences).length === 0;
  });
  if (nextContent.length === message.content.length) {
    return { message, stripped: false };
  }
  return {
    message: { ...message, content: nextContent },
    stripped: true,
  };
}

function userMessageText(message: Extract<AgentMessage, { role: "user" }>): string {
  if (typeof message.content === "string") {
    return message.content.trim();
  }
  return message.content
    .filter(
      (block): block is Extract<(typeof message.content)[number], { type: "text" }> =>
        block?.type === "text" && typeof block.text === "string",
    )
    .map((block) => block.text.trim())
    .filter(Boolean)
    .join("\n\n");
}

export function transcriptTurnsFromMessages(messages: AgentMessage[]): ContextPollutionTurn[] {
  const turns: ContextPollutionTurn[] = [];
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (!message) {
      continue;
    }
    if (message.role === "user") {
      turns.push({
        id: `msg-${index}`,
        role: "user",
        text: userMessageText(message),
        order: index,
        source: "transcript",
      });
      continue;
    }
    if (message.role !== "assistant") {
      continue;
    }
    const text = assistantMessageText(message);
    const removable = !message.content.some((block) => block?.type === "toolCall");
    turns.push({
      id: `msg-${index}`,
      role: "assistant",
      text,
      order: index,
      source: "transcript",
      removable,
    });
  }
  return turns.filter((turn) => turn.text.trim());
}

export function transcriptTurnsFromSessionEntries(
  entries: Array<{
    id?: string;
    type?: string;
    message?: AgentMessage;
  }>,
): ContextPollutionTurn[] {
  const turns: ContextPollutionTurn[] = [];
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    if (entry?.type !== "message" || !entry.message) {
      continue;
    }
    const message = entry.message;
    if (message.role === "user") {
      turns.push({
        id: entry.id ?? `entry-${index}`,
        role: "user",
        text: userMessageText(message),
        order: index,
        source: "transcript",
      });
      continue;
    }
    if (message.role !== "assistant") {
      continue;
    }
    const text = assistantMessageText(message);
    const removable = !message.content.some((block) => block?.type === "toolCall");
    turns.push({
      id: entry.id ?? `entry-${index}`,
      role: "assistant",
      text,
      order: index,
      source: "transcript",
      removable,
    });
  }
  return turns.filter((turn) => turn.text.trim());
}

export function taskContextTurnsFromEntries(entries: TaskContextEntry[]): ContextPollutionTurn[] {
  return entries
    .filter((entry) => entry.role === "user" || entry.role === "assistant")
    .map((entry) => ({
      id: entry.id,
      role: entry.role,
      text: entry.content.trim(),
      order: entry.sequence,
      source: "task-context" as const,
      removable: entry.role === "assistant",
    }))
    .filter((turn) => turn.text.trim());
}

export function filterTranscriptMessagesForReplyPreferences(messages: AgentMessage[]): {
  messages: AgentMessage[];
  removedCount: number;
  removedIndexes: number[];
  sanitizedCount: number;
  sanitizedIndexes: number[];
  scan: ContextPollutionScan;
} {
  const turns = transcriptTurnsFromMessages(messages);
  const scan = scanContextPollution(turns);
  const nextMessages: AgentMessage[] = [];
  let preferences: ReplyFormatPreferences = {
    noPinyinChinese: false,
    noInlineEnglishChinese: false,
  };
  const removedIndexes: number[] = [];
  const sanitizedIndexes: number[] = [];
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (!message) {
      continue;
    }
    if (message.role === "user") {
      preferences = detectReplyFormatPreferencesFromText(userMessageText(message), preferences);
      nextMessages.push(message);
      continue;
    }
    if (message.role !== "assistant" || !Array.isArray(message.content)) {
      nextMessages.push(message);
      continue;
    }
    const reasons = detectContextPollutionReasons(assistantMessageText(message), preferences);
    if (reasons.length === 0) {
      nextMessages.push(message);
      continue;
    }
    const hasToolCall = message.content.some((block) => block?.type === "toolCall");
    if (!hasToolCall) {
      removedIndexes.push(index);
      continue;
    }
    const stripped = stripPollutingAssistantTextBlocks(message, preferences);
    sanitizedIndexes.push(index);
    nextMessages.push(stripped.message);
  }
  return {
    messages: nextMessages,
    removedCount: removedIndexes.length,
    removedIndexes,
    sanitizedCount: sanitizedIndexes.length,
    sanitizedIndexes,
    scan,
  };
}

export function filterTaskContextEntriesForReplyPreferences(entries: TaskContextEntry[]): {
  entries: TaskContextEntry[];
  removedIds: string[];
  scan: ContextPollutionScan;
} {
  const turns = taskContextTurnsFromEntries(entries);
  const scan = scanContextPollution(turns);
  const removedIds = scan.violations
    .filter((violation) => violation.source === "task-context")
    .map((violation) => violation.id);
  if (removedIds.length === 0) {
    return { entries, removedIds: [], scan };
  }
  const removedSet = new Set(removedIds);
  return {
    entries: entries.filter((entry) => !removedSet.has(entry.id)),
    removedIds,
    scan,
  };
}

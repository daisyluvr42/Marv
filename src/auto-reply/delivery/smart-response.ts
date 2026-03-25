import type { SmartResponseConfig } from "../../core/config/types.messages.js";

export type SmartResponseDecision = "respond" | "skip";

type HeuristicResult = "respond" | "skip" | "uncertain";

/**
 * Sliding-window rate limiter for smart auto-responses per group.
 * Prevents the bot from dominating the conversation.
 */
const rateLimitWindows = new Map<string, number[]>();

function isRateLimited(sessionKey: string, maxPerMinute: number): boolean {
  const now = Date.now();
  const windowMs = 60_000;
  let timestamps = rateLimitWindows.get(sessionKey);
  if (!timestamps) {
    timestamps = [];
    rateLimitWindows.set(sessionKey, timestamps);
  }
  // Evict expired entries.
  const cutoff = now - windowMs;
  while (timestamps.length > 0 && timestamps[0] < cutoff) {
    timestamps.shift();
  }
  return timestamps.length >= maxPerMinute;
}

function recordResponse(sessionKey: string): void {
  const timestamps = rateLimitWindows.get(sessionKey) ?? [];
  timestamps.push(Date.now());
  rateLimitWindows.set(sessionKey, timestamps);
}

const NOISE_PATTERNS =
  /^(?:lol|haha|ok|okay|yeah|yep|yup|nah|nope|hmm|wow|nice|cool|sure|thanks|thx|ty|k|lmao|rofl|bruh|omg|ikr|fr|💀|😂|🤣|👍|👎|❤️|🔥|😍|🙄|🤔|💯|😭|😅|🤡|💀💀|😂😂)$/i;

/**
 * Run fast heuristics to decide whether the bot should respond.
 * Returns "respond", "skip", or "uncertain" (needs LLM classifier).
 */
export function classifyWithHeuristics(params: {
  messageBody: string;
  wasMentioned: boolean;
  /** True if the message is a reply to one of the bot's own messages. */
  isReplyToBot: boolean;
  botName?: string;
  senderName?: string;
  /** Recent message history (for detecting rapid human-to-human exchanges). */
  recentHistory?: Array<{ sender: string; body: string }>;
  /** Custom always-respond patterns from config. */
  alwaysRespondPatterns?: string[];
}): HeuristicResult {
  const { messageBody, wasMentioned, isReplyToBot, botName, recentHistory, alwaysRespondPatterns } =
    params;
  const text = messageBody.trim();

  // Direct mention or reply to bot — always respond.
  if (wasMentioned || isReplyToBot) {
    return "respond";
  }

  // Empty message — skip.
  if (!text) {
    return "skip";
  }

  // Noise / reaction messages — skip.
  if (NOISE_PATTERNS.test(text)) {
    return "skip";
  }

  // Bot name mentioned in text — respond.
  if (botName) {
    const namePattern = new RegExp(`\\b${escapeRegex(botName)}\\b`, "i");
    if (namePattern.test(text)) {
      return "respond";
    }
  }

  // Custom always-respond patterns.
  if (alwaysRespondPatterns) {
    for (const pattern of alwaysRespondPatterns) {
      try {
        if (new RegExp(pattern, "i").test(text)) {
          return "respond";
        }
      } catch {
        // Invalid regex — skip.
      }
    }
  }

  // Direct question with "you" — likely directed at someone specific.
  if (text.endsWith("?") && /\byou\b/i.test(text)) {
    return "uncertain";
  }

  // General question — uncertain (let LLM decide).
  if (text.endsWith("?")) {
    return "uncertain";
  }

  // Rapid human-to-human exchange (3+ recent messages, none mentioning bot) — skip.
  if (recentHistory && recentHistory.length >= 3) {
    const lastFew = recentHistory.slice(-3);
    const uniqueSenders = new Set(lastFew.map((m) => m.sender));
    const anyMentionBot =
      botName && lastFew.some((m) => new RegExp(`\\b${escapeRegex(botName)}\\b`, "i").test(m.body));
    if (uniqueSenders.size >= 2 && !anyMentionBot) {
      return "skip";
    }
  }

  return "uncertain";
}

/**
 * Build a classifier prompt for the LLM to decide whether to respond.
 */
export function buildClassifierPrompt(params: {
  botName: string;
  senderName: string;
  messageBody: string;
  recentContext?: Array<{ sender: string; body: string }>;
}): string {
  const { botName, senderName, messageBody, recentContext } = params;
  const contextLines = (recentContext ?? [])
    .slice(-5)
    .map((m) => `  ${m.sender}: ${m.body.slice(0, 200)}`)
    .join("\n");
  return [
    `You are deciding whether "${botName}" should respond to a message in a group chat.`,
    `${botName} is a helpful colleague who responds when addressed, asked questions, or when they can add clear value.`,
    `${botName} should NOT respond to casual side conversations between other people, noise messages, or topics outside their expertise.`,
    "",
    contextLines ? `Recent context:\n${contextLines}\n` : "",
    `Current message from ${senderName}: ${messageBody.slice(0, 500)}`,
    "",
    "Should the assistant respond? Answer exactly RESPOND or SKIP on the first line.",
  ]
    .filter((line) => line !== undefined)
    .join("\n");
}

/**
 * Parse the classifier's response into a decision.
 */
export function parseClassifierResponse(response: string): SmartResponseDecision {
  const firstLine = response.trim().split("\n")[0]?.trim().toUpperCase() ?? "";
  if (firstLine.startsWith("RESPOND")) {
    return "respond";
  }
  if (firstLine.startsWith("SKIP")) {
    return "skip";
  }
  // Default to skip if unclear (err on the side of silence).
  return "skip";
}

/**
 * Full smart response decision pipeline.
 * Returns "respond" or "skip".
 */
export function evaluateSmartResponse(params: {
  messageBody: string;
  wasMentioned: boolean;
  isReplyToBot: boolean;
  botName?: string;
  senderName?: string;
  recentHistory?: Array<{ sender: string; body: string }>;
  sessionKey: string;
  config?: SmartResponseConfig;
  /** LLM classifier result (if heuristics returned "uncertain" and caller ran the classifier). */
  classifierDecision?: SmartResponseDecision;
}): SmartResponseDecision {
  const { config, sessionKey, wasMentioned } = params;
  const maxPerMinute = config?.maxAutoResponsesPerMinute ?? 5;

  // Run heuristics first.
  const heuristic = classifyWithHeuristics({
    messageBody: params.messageBody,
    wasMentioned: params.wasMentioned,
    isReplyToBot: params.isReplyToBot,
    botName: params.botName,
    senderName: params.senderName,
    recentHistory: params.recentHistory,
    alwaysRespondPatterns: config?.alwaysRespondPatterns,
  });

  if (heuristic === "skip") {
    return "skip";
  }

  // Direct mentions bypass rate limiter.
  if (heuristic === "respond" && wasMentioned) {
    recordResponse(sessionKey);
    return "respond";
  }

  // For heuristic "respond" (non-mention) and "uncertain" — check rate limit.
  if (isRateLimited(sessionKey, maxPerMinute)) {
    return "skip";
  }

  // Use classifier decision if available for "uncertain" cases.
  if (heuristic === "uncertain") {
    const decision = params.classifierDecision ?? "skip";
    if (decision === "respond") {
      recordResponse(sessionKey);
    }
    return decision;
  }

  // Heuristic said "respond" (non-mention, not rate-limited).
  recordResponse(sessionKey);
  return "respond";
}

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

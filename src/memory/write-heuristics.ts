export type MemoryWriteHeuristicClassification =
  | "explicit_memory"
  | "durable_preference"
  | "durable_identity_fact"
  | "project_convention"
  | "reject_small_talk"
  | "reject_question"
  | "reject_transient"
  | "reject_unclear";

export type MemoryWriteHeuristicDecision =
  | {
      shouldWrite: true;
      classification:
        | "explicit_memory"
        | "durable_preference"
        | "durable_identity_fact"
        | "project_convention";
      normalizedContent: string;
    }
  | {
      shouldWrite: false;
      classification: Exclude<
        MemoryWriteHeuristicClassification,
        "explicit_memory" | "durable_preference" | "durable_identity_fact" | "project_convention"
      >;
      normalizedContent: string;
    };

const SMALL_TALK_RE =
  /^(hi|hello|hey|yo|thanks|thank you|ok|okay|cool|nice|great|sounds good|good morning|good night|bye|goodbye|你好|嗨|谢谢|好的|再见)[.!? ]*$/i;
const EXPLICIT_MEMORY_PREFIX_RE =
  /^(?:please\s+)?(?:remember|note|keep in mind|save)\s+(?:that\s+|this\s+|for later:\s*)?/i;
const EXPLICIT_MEMORY_PREFIX_ZH_RE = /^(?:请)?(?:记住|记一下|记下来|记着|保存到记忆里?)[:：,\s]*/i;
const EXPLICIT_MEMORY_INLINE_RE =
  /\b(?:remember this|remember that|save this|note this down|keep this in memory)\b/i;
const EXPLICIT_MEMORY_INLINE_ZH_RE = /(?:帮我)?(?:记住|记一下|记下来|存一下|保存到记忆)/i;
const QUESTION_RE =
  /[?？]\s*$|^(?:what|which|when|where|who|why|how|can|could|would|should|do|does|did|is|are|am|will|请问|能不能|可以|是否|怎么|为什么|谁|什么|哪|多少)\b/i;
const TRANSIENT_RE =
  /\b(?:right now|currently|for now|today|tomorrow|this week|temporarily|temporary|tmp|investigating|debugging|working on|trying to|in progress|just for this task)\b/i;
const TRANSIENT_ZH_RE = /(?:现在|目前|今天|明天|暂时|临时|正在|调试|排查|这次任务|这一轮)/i;
const PREFERENCE_RE =
  /\b(?:i prefer|my preference is|prefer to|please always|always use|never use|do not use|don't use)\b/i;
const PREFERENCE_ZH_RE = /(?:我偏好|我喜欢|请始终|请一直|总是用|不要用|别用)/i;
const IDENTITY_RE = /\b(?:i am|i'm|my name is|i work on|i live in|my timezone is|my role is)\b/i;
const IDENTITY_ZH_RE = /(?:我是|我的名字是|我负责|我在.*工作|我的时区是)/i;
const PROJECT_RE =
  /\b(?:policy|guardrail|principle|standard|convention|workflow|runbook|playbook|checklist)\b/i;
const PROJECT_ZH_RE = /(?:规范|约定|流程|守则|原则|策略|清单|运行手册)/i;

const DURABLE_KINDS = new Set(["preference", "identity", "policy", "guardrail", "principle"]);

function normalizeWhitespace(input: string): string {
  return input.replace(/\s+/g, " ").trim();
}

function stripExplicitMemoryPrefixes(input: string): string {
  let next = input.trim();
  next = next.replace(EXPLICIT_MEMORY_PREFIX_RE, "");
  next = next.replace(EXPLICIT_MEMORY_PREFIX_ZH_RE, "");
  return next.trim();
}

function isQuestionLike(input: string): boolean {
  return QUESTION_RE.test(input);
}

function isSmallTalk(input: string): boolean {
  return SMALL_TALK_RE.test(input);
}

function isTransient(input: string): boolean {
  return TRANSIENT_RE.test(input) || TRANSIENT_ZH_RE.test(input);
}

function classifyDurableSignal(params: {
  normalized: string;
  lowered: string;
  kind: string;
}): "durable_preference" | "durable_identity_fact" | "project_convention" | "reject_unclear" {
  const { normalized, lowered, kind } = params;
  if (kind === "preference" || PREFERENCE_RE.test(lowered) || PREFERENCE_ZH_RE.test(normalized)) {
    return "durable_preference";
  }
  if (kind === "identity" || IDENTITY_RE.test(lowered) || IDENTITY_ZH_RE.test(normalized)) {
    return "durable_identity_fact";
  }
  if (
    kind === "policy" ||
    kind === "guardrail" ||
    kind === "principle" ||
    PROJECT_RE.test(lowered) ||
    PROJECT_ZH_RE.test(normalized)
  ) {
    return "project_convention";
  }
  if (DURABLE_KINDS.has(kind)) {
    return "project_convention";
  }
  return "reject_unclear";
}

export function evaluateMemoryWriteHeuristics(params: {
  content: string;
  kind?: string;
}): MemoryWriteHeuristicDecision {
  const normalized = normalizeWhitespace(params.content);
  const lowered = normalized.toLowerCase();
  const kind = normalizeWhitespace(params.kind ?? "note").toLowerCase();

  if (!normalized || normalized.length < 8) {
    return {
      shouldWrite: false,
      classification: "reject_unclear",
      normalizedContent: normalized,
    };
  }

  const hasExplicitCue =
    EXPLICIT_MEMORY_PREFIX_RE.test(normalized) ||
    EXPLICIT_MEMORY_PREFIX_ZH_RE.test(normalized) ||
    EXPLICIT_MEMORY_INLINE_RE.test(lowered) ||
    EXPLICIT_MEMORY_INLINE_ZH_RE.test(normalized);
  if (hasExplicitCue) {
    const stripped = normalizeWhitespace(stripExplicitMemoryPrefixes(normalized));
    return {
      shouldWrite: true,
      classification: "explicit_memory",
      normalizedContent: stripped || normalized,
    };
  }

  if (isSmallTalk(normalized)) {
    return {
      shouldWrite: false,
      classification: "reject_small_talk",
      normalizedContent: normalized,
    };
  }

  if (isQuestionLike(normalized)) {
    return {
      shouldWrite: false,
      classification: "reject_question",
      normalizedContent: normalized,
    };
  }

  const durableClassification = classifyDurableSignal({ normalized, lowered, kind });
  if (durableClassification !== "reject_unclear") {
    return {
      shouldWrite: true,
      classification: durableClassification,
      normalizedContent: normalized,
    };
  }

  if (isTransient(normalized)) {
    return {
      shouldWrite: false,
      classification: "reject_transient",
      normalizedContent: normalized,
    };
  }

  return {
    shouldWrite: false,
    classification: "reject_unclear",
    normalizedContent: normalized,
  };
}

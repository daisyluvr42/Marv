import type { MarvConfig } from "../config/config.js";
import type {
  AutoRoutingComplexity,
  AutoRoutingConfig,
  AutoRoutingRule,
  AutoRoutingThresholds,
} from "../config/types.agent-defaults.js";
import { resolveAgentConfig } from "./agent-scope.js";
import type { ThinkLevel } from "./model/model-selection.js";
import { parseModelRef } from "./model/model-selection.js";

export type AutoRoutingResult = {
  complexity: AutoRoutingComplexity;
  provider?: string;
  model?: string;
  thinking?: ThinkLevel;
  /** True when auto-routing was active and matched a rule. */
  routed: boolean;
};

// -- Thresholds defaults --

const DEFAULT_SIMPLE_MAX_CHARS = 200;
const DEFAULT_MODERATE_MAX_CHARS = 600;
const DEFAULT_COMPLEX_MAX_CHARS = 1500;

const DEFAULT_COMPLEX_PATTERNS = [
  "\\b(implement|refactor|architect|design|debug|optimize)\\b",
  "\\b(analyze|compare|evaluate|research|investigate)\\b",
];

// -- Heuristic signals --

const CODE_MARKERS = [
  /```/,
  /\bfunction\s/,
  /\bclass\s/,
  /\bimport\s/,
  /\bexport\s/,
  /\bconst\s/,
  /\bdef\s/,
  /\basync\s/,
  /\bawait\s/,
  /=>/, // arrow functions
  /\bif\s*\(/,
  /\bfor\s*\(/,
  /\breturn\s/,
];

const MULTI_PART_MARKERS = [
  /\d+[.)]\s/, // numbered lists
  /\band\s+then\b/i,
  /\balso\b/i,
  /\bfirst\b.*\bthen\b/is,
  /\bstep\s+\d/i,
];

function countMatches(text: string, patterns: RegExp[]): number {
  let count = 0;
  for (const pattern of patterns) {
    if (pattern.test(text)) {
      count += 1;
    }
  }
  return count;
}

function compilePatterns(raw: string[]): RegExp[] {
  const compiled: RegExp[] = [];
  for (const pattern of raw) {
    try {
      compiled.push(new RegExp(pattern, "i"));
    } catch {
      // Skip invalid patterns silently.
    }
  }
  return compiled;
}

/**
 * Classify message complexity using heuristic rules.
 */
export function classifyComplexityByRules(params: {
  prompt: string;
  hasImages?: boolean;
  thresholds?: AutoRoutingThresholds;
}): AutoRoutingComplexity {
  const { prompt, hasImages } = params;
  const t = params.thresholds;
  const simpleMax = t?.simpleMaxChars ?? DEFAULT_SIMPLE_MAX_CHARS;
  const moderateMax = t?.moderateMaxChars ?? DEFAULT_MODERATE_MAX_CHARS;
  const complexMax = t?.complexMaxChars ?? DEFAULT_COMPLEX_MAX_CHARS;
  const complexPatternStrings = t?.complexPatterns ?? DEFAULT_COMPLEX_PATTERNS;

  const length = prompt.length;
  const codeHits = countMatches(prompt, CODE_MARKERS);
  const multiPartHits = countMatches(prompt, MULTI_PART_MARKERS);
  const complexPatterns = compilePatterns(complexPatternStrings);
  const complexPatternHits = countMatches(prompt, complexPatterns);

  // Score-based approach: accumulate complexity signals.
  let score = 0;

  // Length signal
  if (length >= complexMax) {
    score += 4;
  } else if (length >= moderateMax) {
    score += 2;
  } else if (length >= simpleMax) {
    score += 1;
  }

  // Code markers
  if (codeHits >= 4) {
    score += 3;
  } else if (codeHits >= 2) {
    score += 2;
  } else if (codeHits >= 1) {
    score += 1;
  }

  // Complex pattern keywords
  if (complexPatternHits >= 2) {
    score += 2;
  } else if (complexPatternHits >= 1) {
    score += 1;
  }

  // Multi-part requests
  if (multiPartHits >= 2) {
    score += 2;
  } else if (multiPartHits >= 1) {
    score += 1;
  }

  // Images increase complexity
  if (hasImages) {
    score += 1;
  }

  // Map score to complexity tier
  if (score >= 6) {
    return "expert";
  }
  if (score >= 4) {
    return "complex";
  }
  if (score >= 2) {
    return "moderate";
  }
  return "simple";
}

/**
 * Build the LLM classification prompt for a lightweight model.
 */
export function buildClassifierPrompt(userMessage: string): string {
  const truncated = userMessage.length > 2000 ? userMessage.slice(0, 2000) + "..." : userMessage;
  return [
    "Classify the following user message into exactly one complexity tier.",
    "Reply with ONLY one word: simple, moderate, complex, or expert.",
    "",
    "Tiers:",
    "- simple: greetings, short factual questions, simple lookups",
    "- moderate: questions needing explanation, short code snippets, single-step tasks",
    "- complex: multi-step tasks, debugging, code review, analysis",
    "- expert: architecture design, large refactoring, research, multi-file implementation",
    "",
    "User message:",
    truncated,
  ].join("\n");
}

const VALID_COMPLEXITIES = new Set<AutoRoutingComplexity>([
  "simple",
  "moderate",
  "complex",
  "expert",
]);

/**
 * Parse the LLM classifier response into a complexity tier.
 * Falls back to "moderate" if parsing fails.
 */
export function parseClassifierResponse(response: string): AutoRoutingComplexity {
  const trimmed = response.trim().toLowerCase();
  // Try exact match first
  if (VALID_COMPLEXITIES.has(trimmed as AutoRoutingComplexity)) {
    return trimmed as AutoRoutingComplexity;
  }
  // Try to find a tier word in the response
  for (const tier of VALID_COMPLEXITIES) {
    if (trimmed.includes(tier)) {
      return tier;
    }
  }
  return "moderate";
}

/**
 * Find the matching routing rule for a given complexity.
 */
function findMatchingRule(
  rules: AutoRoutingRule[] | undefined,
  complexity: AutoRoutingComplexity,
): AutoRoutingRule | undefined {
  if (!rules || rules.length === 0) {
    return undefined;
  }
  return rules.find((r) => r.complexity === complexity);
}

/**
 * Resolve the auto-routing config for an agent, with per-agent override > global defaults.
 */
function resolveAutoRoutingConfig(
  config: MarvConfig | undefined,
  agentId?: string,
): AutoRoutingConfig | undefined {
  if (!config) {
    return undefined;
  }
  // Per-agent override
  if (agentId) {
    const agentConfig = resolveAgentConfig(config, agentId);
    if (agentConfig?.autoRouting) {
      return agentConfig.autoRouting;
    }
  }
  return config.agents?.defaults?.autoRouting;
}

/**
 * Top-level auto-routing entry point.
 * Returns AutoRoutingResult with routed=false when disabled or no matching rule.
 *
 * For classifier: "llm", the caller must provide classifyFn (async function that
 * sends a prompt to the classifier model and returns the response text).
 * This keeps the auto-routing module decoupled from the provider/model infra.
 */
export async function resolveAutoRouting(params: {
  prompt: string;
  hasImages?: boolean;
  config?: MarvConfig;
  agentId?: string;
  defaultProvider: string;
  defaultModel: string;
  /** Optional LLM classify function (injected by caller for classifier: "llm"). */
  classifyFn?: (classifierPrompt: string) => Promise<string>;
}): Promise<AutoRoutingResult> {
  const routingConfig = resolveAutoRoutingConfig(params.config, params.agentId);

  if (!routingConfig?.enabled) {
    return { complexity: "simple", routed: false };
  }

  const classifier = routingConfig.classifier ?? "rules";
  let complexity: AutoRoutingComplexity;

  if (classifier === "llm" && params.classifyFn) {
    try {
      const classifierPrompt = buildClassifierPrompt(params.prompt);
      const response = await params.classifyFn(classifierPrompt);
      complexity = parseClassifierResponse(response);
    } catch {
      // Fall back to rules on LLM failure.
      complexity = classifyComplexityByRules({
        prompt: params.prompt,
        hasImages: params.hasImages,
        thresholds: routingConfig.thresholds,
      });
    }
  } else {
    complexity = classifyComplexityByRules({
      prompt: params.prompt,
      hasImages: params.hasImages,
      thresholds: routingConfig.thresholds,
    });
  }

  const rule = findMatchingRule(routingConfig.rules, complexity);
  if (!rule) {
    return { complexity, routed: false };
  }

  const parsed = parseModelRef(rule.model, params.defaultProvider);
  if (!parsed) {
    return { complexity, routed: false };
  }

  return {
    complexity,
    provider: parsed.provider,
    model: parsed.model,
    thinking: rule.thinking as ThinkLevel | undefined,
    routed: true,
  };
}

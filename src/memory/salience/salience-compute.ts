import type {
  ClarityDecayConfig,
  FusionWeights,
  ScopePenaltyConfig,
  SoulMemoryTierValue,
  TierPriorityConfig,
} from "./salience-types.js";

export const P0_SCOPE_PENALTY = 0.8;
export const CROSS_SCOPE_PENALTY = 0.2;
export const MATCH_SCOPE_PENALTY = 1;

export const P0_TIER_MULTIPLIER = 1.2;
export const P1_TIER_MULTIPLIER = 1;
export const P2_TIER_MULTIPLIER = 0.75;
export const P3_TIER_MULTIPLIER = 0.3;

export const SCORE_SIMILARITY_WEIGHT = 1;
export const SCORE_DECAY_WEIGHT = 1;

export const FORGET_CONFIDENCE_THRESHOLD = 0.1;
export const FORGET_STREAK_HALF_LIVES = 3;
export const P0_CLARITY_HALF_LIFE_DAYS = 365;
export const P1_CLARITY_HALF_LIFE_DAYS = 45;
export const P2_CLARITY_HALF_LIFE_DAYS = 10;
export const P3_CLARITY_HALF_LIFE_DAYS = 3;

export const FUSION_VECTOR_WEIGHT = 0.32;
export const FUSION_LEXICAL_WEIGHT = 0.15;
export const FUSION_BM25_WEIGHT = 0.15;
export const FUSION_GRAPH_WEIGHT = 0.16;
export const FUSION_CLUSTER_WEIGHT = 0.06;

const DEFAULT_FUSION_WEIGHTS: FusionWeights = {
  vector: FUSION_VECTOR_WEIGHT,
  lexical: FUSION_LEXICAL_WEIGHT,
  bm25: FUSION_BM25_WEIGHT,
  graph: FUSION_GRAPH_WEIGHT,
  cluster: FUSION_CLUSTER_WEIGHT,
};

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

function resolveTierHalfLifeDays(
  tier: SoulMemoryTierValue,
  config: ClarityDecayConfig,
): number | null {
  if (tier === "P1") {
    return config.p1ClarityHalfLifeDays;
  }
  if (tier === "P2") {
    return config.p2ClarityHalfLifeDays;
  }
  if (tier === "P3") {
    return config.p3ClarityHalfLifeDays;
  }
  return null;
}

function computeBelowThresholdDurationDays(params: {
  item: { confidence: number };
  ageDays: number;
  threshold: number;
  halfLifeDays: number;
}): number {
  const ageDays = Math.max(0, params.ageDays);
  const threshold = clamp(params.threshold, 0, 1);
  const baseConfidence = clamp(params.item.confidence, 0, 1);
  if (baseConfidence <= 0 || baseConfidence <= threshold) {
    return ageDays;
  }
  const crossingAgeDays = params.halfLifeDays * Math.log2(baseConfidence / threshold);
  if (!Number.isFinite(crossingAgeDays) || crossingAgeDays <= 0) {
    return ageDays;
  }
  return Math.max(0, ageDays - crossingAgeDays);
}

export function computeFusionSemanticMatch(
  params: {
    vectorScore: number;
    lexicalScore: number;
    bm25Score: number;
    graphScore: number;
    clusterScore: number;
  },
  weights: FusionWeights = DEFAULT_FUSION_WEIGHTS,
): number {
  const weightSum =
    weights.vector + weights.lexical + weights.bm25 + weights.graph + weights.cluster;
  if (weightSum <= 0) {
    return 0;
  }
  const weighted =
    params.vectorScore * weights.vector +
    params.lexicalScore * weights.lexical +
    params.bm25Score * weights.bm25 +
    params.graphScore * weights.graph +
    params.clusterScore * weights.cluster;
  return clamp(weighted / weightSum, 0, 1);
}

export function computeWeightedScore(value: number, weight: number): number {
  const normalizedValue = clamp(value, 0, 1);
  if (!Number.isFinite(weight) || weight === 1) {
    return normalizedValue;
  }
  return clamp(normalizedValue ** weight, 0, 1);
}

export function tierPriorityFactor(tier: SoulMemoryTierValue, config: TierPriorityConfig): number {
  if (tier === "P0") {
    return config.p0TierMultiplier;
  }
  if (tier === "P2") {
    return config.p2TierMultiplier;
  }
  if (tier === "P3") {
    return config.p3TierMultiplier;
  }
  return config.p1TierMultiplier;
}

export function resolveScopePenalty(
  params: {
    item: { scopeType: string; scopeId: string };
    activeScopeKeySet: Set<string>;
  },
  config: ScopePenaltyConfig,
): number {
  if (params.activeScopeKeySet.has(`${params.item.scopeType}:${params.item.scopeId}`)) {
    return config.matchScopePenalty;
  }
  if (params.item.scopeType === "global" || params.item.scopeType === "user") {
    return config.p0ScopePenalty;
  }
  return config.crossScopePenalty;
}

export function clarityDecayFactor(
  tier: SoulMemoryTierValue,
  ageDays: number,
  config: ClarityDecayConfig,
): number {
  const normalizedAgeDays = Math.max(0, ageDays);
  const halfLifeDays =
    tier === "P0"
      ? config.p0ClarityHalfLifeDays
      : tier === "P2"
        ? config.p2ClarityHalfLifeDays
        : tier === "P3"
          ? config.p3ClarityHalfLifeDays
          : config.p1ClarityHalfLifeDays;
  const factor = 0.5 ** (normalizedAgeDays / halfLifeDays);
  return clamp(factor, 0, 1);
}

export function computeCurrentClarity(
  item: { confidence: number; tier: SoulMemoryTierValue },
  ageDays: number,
  config: ClarityDecayConfig,
): number {
  return clamp(item.confidence * clarityDecayFactor(item.tier, ageDays, config), 0, 1);
}

export function shouldPruneMemoryItem(
  item: { confidence: number; tier: SoulMemoryTierValue },
  ageDays: number,
  config: ClarityDecayConfig,
): boolean {
  const halfLifeDays = resolveTierHalfLifeDays(item.tier, config);
  if (!halfLifeDays || !Number.isFinite(halfLifeDays) || halfLifeDays <= 0) {
    return false;
  }
  const clarity = computeCurrentClarity(item, ageDays, config);
  if (clarity >= config.forgetConfidenceThreshold) {
    return false;
  }
  const belowThresholdDays = computeBelowThresholdDurationDays({
    item,
    ageDays,
    threshold: config.forgetConfidenceThreshold,
    halfLifeDays,
  });
  return belowThresholdDays >= halfLifeDays * config.forgetStreakHalfLives;
}

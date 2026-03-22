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

/** @deprecated Tier multipliers are no longer used. All items are P3. Kept for backward compat. */
export const P0_TIER_MULTIPLIER = 1;
/** @deprecated */
export const P1_TIER_MULTIPLIER = 1;
/** @deprecated */
export const P2_TIER_MULTIPLIER = 1;
/** @deprecated */
export const P3_TIER_MULTIPLIER = 1;

export const SCORE_SIMILARITY_WEIGHT = 1;
export const SCORE_DECAY_WEIGHT = 1;

export const FORGET_CONFIDENCE_THRESHOLD = 0.1;
export const FORGET_STREAK_HALF_LIVES = 3;
/** @deprecated Clarity decay is no longer applied. Kept for backward compat. */
export const P0_CLARITY_HALF_LIFE_DAYS = Infinity;
/** @deprecated */
export const P1_CLARITY_HALF_LIFE_DAYS = Infinity;
/** @deprecated */
export const P2_CLARITY_HALF_LIFE_DAYS = Infinity;
/** @deprecated */
export const P3_CLARITY_HALF_LIFE_DAYS = Infinity;

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

/** @deprecated All items are P3; tier multiplier is always 1. */
export function tierPriorityFactor(
  _tier: SoulMemoryTierValue,
  _config: TierPriorityConfig,
): number {
  return 1;
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

/** @deprecated No decay applied in the new architecture. Always returns 1. */
export function clarityDecayFactor(
  _tier: SoulMemoryTierValue,
  _ageDays: number,
  _config: ClarityDecayConfig,
): number {
  return 1;
}

/** @deprecated No decay applied. Returns confidence directly. */
export function computeCurrentClarity(
  item: { confidence: number; tier: SoulMemoryTierValue },
  _ageDays: number,
  _config: ClarityDecayConfig,
): number {
  return clamp(item.confidence, 0, 1);
}

/** @deprecated No decay-based pruning in the new architecture. Always returns false. */
export function shouldPruneMemoryItem(
  _item: { confidence: number; tier: SoulMemoryTierValue },
  _ageDays: number,
  _config: ClarityDecayConfig,
): boolean {
  return false;
}

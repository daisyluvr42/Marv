import type { FusionWeights, ScopePenaltyConfig } from "./salience-types.js";

export const GLOBAL_SCOPE_PENALTY = 0.8;
export const CROSS_SCOPE_PENALTY = 0.2;
export const MATCH_SCOPE_PENALTY = 1;

export const SCORE_SIMILARITY_WEIGHT = 1;

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
    return config.globalScopePenalty;
  }
  return config.crossScopePenalty;
}

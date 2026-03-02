export type SoulMemoryTierValue = "P0" | "P1" | "P2";

export type FusionWeights = {
  vector: number;
  lexical: number;
  bm25: number;
  graph: number;
  cluster: number;
};

export type SalienceWeights = {
  scoreSimilarityWeight: number;
  scoreDecayWeight: number;
  reinforcementLogWeight: number;
};

export type SalienceScoreBreakdown = {
  relevanceScore: number;
  similarityScore: number;
  decayFactor: number;
  decayScore: number;
  reinforcementFactor: number;
  salienceDecay: number;
  salienceReinforcement: number;
  salienceScore: number;
};

export type ScopePenaltyConfig = {
  p0ScopePenalty: number;
  crossScopePenalty: number;
  matchScopePenalty: number;
};

export type TierPriorityConfig = {
  p0TierMultiplier: number;
  p1TierMultiplier: number;
  p2TierMultiplier: number;
};

export type ClarityDecayConfig = {
  p0ClarityHalfLifeDays: number;
  p1ClarityHalfLifeDays: number;
  p2ClarityHalfLifeDays: number;
  forgetConfidenceThreshold: number;
  forgetStreakHalfLives: number;
};

export type ReferenceExpansionConfig = {
  referenceExpansionEnabled: boolean;
  referenceMaxHops: number;
  referenceEdgeDecay: number;
  referenceBoostWeight: number;
  referenceMaxBoost: number;
  referenceSeedTopKMultiplier: number;
};

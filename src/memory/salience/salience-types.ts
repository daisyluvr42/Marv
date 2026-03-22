export type FusionWeights = {
  vector: number;
  lexical: number;
  bm25: number;
  graph: number;
  cluster: number;
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
  globalScopePenalty: number;
  crossScopePenalty: number;
  matchScopePenalty: number;
};

export type ReferenceExpansionConfig = {
  referenceExpansionEnabled: boolean;
  referenceMaxHops: number;
  referenceEdgeDecay: number;
  referenceBoostWeight: number;
  referenceMaxBoost: number;
  referenceSeedTopKMultiplier: number;
};

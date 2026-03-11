import type { SessionSendPolicyConfig } from "./types.base.js";

export type MemoryBackend = "builtin" | "qmd";
export type MemoryCitationsMode = "auto" | "on" | "off";
export type MemoryQmdSearchMode = "query" | "search" | "vsearch";

export type MemoryAutoRecallConfig = {
  enabled?: boolean;
  maxResults?: number;
  minScore?: number;
  maxContextChars?: number;
  includeConversationContext?: boolean;
};

export type MemoryKnowledgeVaultConfig = {
  path: string;
  name?: string;
  exclude?: string[];
};

export type MemoryKnowledgeConfig = {
  enabled?: boolean;
  autoSyncOnSearch?: boolean;
  autoSyncOnBoot?: boolean;
  syncIntervalMs?: number;
  vaults?: MemoryKnowledgeVaultConfig[];
};

export type DeepConsolidationModelApi = "ollama" | "openai-completions";

export type DeepConsolidationModelConfig = {
  provider?: string;
  api?: DeepConsolidationModelApi;
  model?: string;
  baseUrl?: string;
  timeoutMs?: number;
};

export type DeepConsolidationConfig = {
  enabled?: boolean;
  schedule?: string;
  maxItems?: number;
  maxReflections?: number;
  clusterSummarization?: boolean;
  conflictJudgment?: boolean;
  crossScopeReflection?: boolean;
  model?: DeepConsolidationModelConfig;
};

export type MemoryConfig = {
  backend?: MemoryBackend;
  citations?: MemoryCitationsMode;
  /** Enable automatic runtime message ingestion into P3. Default: true. */
  runtimeIngest?: boolean;
  p0AllowedKinds?: string[];
  soul?: MemorySoulConfig;
  autoRecall?: MemoryAutoRecallConfig;
  knowledge?: MemoryKnowledgeConfig;
  qmd?: MemoryQmdConfig;
};

export type MemorySoulConfig = {
  p0AllowedKinds?: string[];
  forgetConfidenceThreshold?: number;
  forgetStreakHalfLives?: number;
  p0ClarityHalfLifeDays?: number;
  p1ClarityHalfLifeDays?: number;
  p2ClarityHalfLifeDays?: number;
  p3ClarityHalfLifeDays?: number;
  p0RecallRelevanceThreshold?: number;
  p2ToP1MinClarity?: number;
  p2ToP1MinAgeDays?: number;
  p2ToP1MinScopeCount?: number;
  p1ToP0MinClarity?: number;
  p1ToP0MinAgeDays?: number;
  p0ScopePenalty?: number;
  crossScopePenalty?: number;
  matchScopePenalty?: number;
  p0TierMultiplier?: number;
  p1TierMultiplier?: number;
  p2TierMultiplier?: number;
  p3TierMultiplier?: number;
  scoreSimilarityWeight?: number;
  scoreDecayWeight?: number;
  reinforcementLogWeight?: number;
  referenceExpansionEnabled?: boolean;
  referenceMaxHops?: number;
  referenceEdgeDecay?: number;
  referenceBoostWeight?: number;
  referenceMaxBoost?: number;
  referenceSeedTopKMultiplier?: number;
  deepConsolidation?: DeepConsolidationConfig;
};

export type MemoryQmdConfig = {
  command?: string;
  searchMode?: MemoryQmdSearchMode;
  includeDefaultMemory?: boolean;
  paths?: MemoryQmdIndexPath[];
  sessions?: MemoryQmdSessionConfig;
  update?: MemoryQmdUpdateConfig;
  limits?: MemoryQmdLimitsConfig;
  scope?: SessionSendPolicyConfig;
};

export type MemoryQmdIndexPath = {
  path: string;
  name?: string;
  pattern?: string;
};

export type MemoryQmdSessionConfig = {
  enabled?: boolean;
  exportDir?: string;
  retentionDays?: number;
};

export type MemoryQmdUpdateConfig = {
  interval?: string;
  debounceMs?: number;
  onBoot?: boolean;
  waitForBootSync?: boolean;
  embedInterval?: string;
  commandTimeoutMs?: number;
  updateTimeoutMs?: number;
  embedTimeoutMs?: number;
};

export type MemoryQmdLimitsConfig = {
  maxResults?: number;
  maxSnippetChars?: number;
  maxInjectedChars?: number;
  timeoutMs?: number;
};

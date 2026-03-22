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

export type ExperienceConfig = {
  /** Enable the experience evolution system. Default: true. */
  enabled?: boolean;
  /** Model for LLM distillation. Defaults to agent's mid-tier model. */
  distillerModel?: string;
  /** Model for weekly calibration. Defaults to highest-tier model. */
  calibrationModel?: string;
  /** Model for context distillation. Defaults to local/cheap model. */
  contextModel?: string;
  /** EXPERIENCE.md character budget. Default: 800. */
  experienceBudgetChars?: number;
  /** CONTEXT.md character budget. Default: 400. */
  contextBudgetChars?: number;
  /** Distillation debounce interval in ms. Default: 4h (14400000). */
  distillDebounceMs?: number;
  /** Calibration cron schedule. Default: weekly. */
  calibrationCron?: string;
  /** Experience log retention in days. Default: unlimited. */
  logRetentionDays?: number;
  /** Enable experience attribution tracking. Default: true. */
  attributionEnabled?: boolean;
  /** Model for attribution confirmation. Defaults to local/cheap model. */
  attributionModel?: string;
  /** Days after which an a:0 experience is considered a zombie. Default: 30. */
  zombieAgeDays?: number;
  /** p/a ratio threshold below which an experience is flagged as harmful. Default: 0.3. */
  harmfulRatioThreshold?: number;
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
  /** Experience evolution system configuration. */
  experience?: ExperienceConfig;
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

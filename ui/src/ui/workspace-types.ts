import type { CronRunLogEntry } from "./types.js";

export type WorkspaceSummarySnapshot = {
  startDate: string;
  endDate: string;
  sessionsTouched: number;
  activeDays: number;
  totalTokens: number;
  totalCost: number;
  pendingProactive: number;
  urgentProactive: number;
  nextWakeAtMs: number | null;
  failingJobs: number;
};

export type WorkspaceMemoryScope = {
  scopeType: string;
  scopeId: string;
  weight: number;
};

export type WorkspaceMemoryItem = {
  id: string;
  scopeType: string;
  scopeId: string;
  kind: string;
  content: string;
  summary?: string;
  confidence: number;
  tier: string;
  source: "core_preference" | "manual_log" | "migration" | "auto_extraction" | "runtime_event";
  recordKind: "fact" | "relationship" | "experience" | "soul";
  metadata?: Record<string, unknown>;
  createdAt: number;
  lastAccessedAt?: number;
  reinforcementCount: number;
  lastReinforcedAt?: number;
};

export type WorkspaceMemorySearchItem = WorkspaceMemoryItem & {
  score: number;
  vectorScore: number;
  lexicalScore: number;
  bm25Score: number;
  rrfScore: number;
  graphScore: number;
  clusterScore: number;
  relevanceScore: number;
  scopePenalty: number;
  clarityScore: number;
  tierMultiplier?: number;
  wasRecallBoosted: boolean;
  timeDecay: number;
  salienceScore: number;
  salienceDecay: number;
  salienceReinforcement: number;
  reinforcementFactor: number;
  referenceBoost: number;
  references: string[];
  ageDays: number;
};

export type WorkspaceMemoryListResult = {
  agentId: string;
  items: WorkspaceMemoryItem[];
};

export type WorkspaceMemorySearchResult = {
  agentId: string;
  query: string;
  scopes: WorkspaceMemoryScope[];
  items: WorkspaceMemorySearchItem[];
};

export type WorkspaceDocumentRoot = {
  id: string;
  agentId: string;
  agentIds: string[];
  label: string;
  path: string;
  fileCount: number;
};

export type WorkspaceDocumentEntry = {
  rootId: string;
  agentId: string;
  agentIds: string[];
  relativePath: string;
  name: string;
  category: string;
  extension: string;
  size: number;
  mtimeMs: number;
  preview?: string;
};

export type WorkspaceDocumentsListResult = {
  updatedAt: number;
  roots: WorkspaceDocumentRoot[];
  items: WorkspaceDocumentEntry[];
};

export type WorkspaceDocumentsReadResult = {
  rootId: string;
  agentId: string;
  agentIds: string[];
  relativePath: string;
  name: string;
  size: number;
  mtimeMs: number;
  content: string;
  truncated: boolean;
};

export type WorkspaceCalendarTopSession = {
  key: string;
  label?: string;
  agentId?: string;
  tokens: number;
  cost: number;
  messages: number;
  lastActivity?: number;
};

export type WorkspaceCalendarRun = CronRunLogEntry & {
  jobName?: string;
  agentId?: string;
};

export type WorkspaceCalendarDay = {
  date: string;
  tokens: number;
  cost: number;
  messages: number;
  toolCalls: number;
  errors: number;
  sessionCount: number;
  topSessions: WorkspaceCalendarTopSession[];
  cronRuns: WorkspaceCalendarRun[];
};

export type WorkspaceCalendarSnapshot = {
  startDate: string;
  endDate: string;
  days: WorkspaceCalendarDay[];
};

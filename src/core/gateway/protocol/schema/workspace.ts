import { Type } from "@sinclair/typebox";
import { NonEmptyString } from "./primitives.js";

const MemoryTierSchema = Type.Union([
  Type.Literal("P0"),
  Type.Literal("P1"),
  Type.Literal("P2"),
  Type.Literal("P3"),
]);

const MemorySourceSchema = Type.Union([
  Type.Literal("core_preference"),
  Type.Literal("manual_log"),
  Type.Literal("migration"),
  Type.Literal("auto_extraction"),
  Type.Literal("runtime_event"),
]);

const MemoryRecordKindSchema = Type.Union([
  Type.Literal("fact"),
  Type.Literal("relationship"),
  Type.Literal("experience"),
  Type.Literal("soul"),
]);

export const MemoryScopeSchema = Type.Object(
  {
    scopeType: NonEmptyString,
    scopeId: NonEmptyString,
    weight: Type.Number({ exclusiveMinimum: 0 }),
  },
  { additionalProperties: false },
);

const MemoryItemFields = {
  id: NonEmptyString,
  scopeType: NonEmptyString,
  scopeId: NonEmptyString,
  kind: NonEmptyString,
  content: Type.String(),
  summary: Type.Optional(Type.String()),
  confidence: Type.Number(),
  tier: MemoryTierSchema,
  source: MemorySourceSchema,
  recordKind: MemoryRecordKindSchema,
  metadata: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
  createdAt: Type.Integer({ minimum: 0 }),
  lastAccessedAt: Type.Optional(Type.Integer({ minimum: 0 })),
  reinforcementCount: Type.Integer({ minimum: 0 }),
  lastReinforcedAt: Type.Optional(Type.Integer({ minimum: 0 })),
};

export const MemoryItemSchema = Type.Object(MemoryItemFields, {
  additionalProperties: false,
});

export const MemorySearchItemSchema = Type.Object(
  {
    ...MemoryItemFields,
    score: Type.Number(),
    vectorScore: Type.Number(),
    lexicalScore: Type.Number(),
    bm25Score: Type.Number(),
    rrfScore: Type.Number(),
    graphScore: Type.Number(),
    clusterScore: Type.Number(),
    relevanceScore: Type.Number(),
    scopePenalty: Type.Number(),
    clarityScore: Type.Number(),
    tierMultiplier: Type.Number(),
    wasRecallBoosted: Type.Boolean(),
    timeDecay: Type.Number(),
    salienceScore: Type.Number(),
    salienceDecay: Type.Number(),
    salienceReinforcement: Type.Number(),
    reinforcementFactor: Type.Number(),
    referenceBoost: Type.Number(),
    references: Type.Array(NonEmptyString),
    ageDays: Type.Number({ minimum: 0 }),
  },
  { additionalProperties: false },
);

export const MemoryListParamsSchema = Type.Object(
  {
    agentId: Type.Optional(NonEmptyString),
    sessionKey: Type.Optional(NonEmptyString),
    scopeType: Type.Optional(NonEmptyString),
    scopeId: Type.Optional(NonEmptyString),
    kind: Type.Optional(NonEmptyString),
    tier: Type.Optional(MemoryTierSchema),
    recordKind: Type.Optional(MemoryRecordKindSchema),
    limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 500 })),
  },
  { additionalProperties: false },
);

export const MemorySearchParamsSchema = Type.Object(
  {
    agentId: Type.Optional(NonEmptyString),
    sessionKey: Type.Optional(NonEmptyString),
    query: NonEmptyString,
    topK: Type.Optional(Type.Integer({ minimum: 1, maximum: 100 })),
    minScore: Type.Optional(Type.Number({ minimum: 0, maximum: 2 })),
    ttlDays: Type.Optional(Type.Integer({ minimum: 0, maximum: 3650 })),
    scopes: Type.Optional(Type.Array(MemoryScopeSchema)),
  },
  { additionalProperties: false },
);

export const MemoryListResultSchema = Type.Object(
  {
    agentId: NonEmptyString,
    items: Type.Array(MemoryItemSchema),
  },
  { additionalProperties: false },
);

export const MemorySearchResultSchema = Type.Object(
  {
    agentId: NonEmptyString,
    query: NonEmptyString,
    scopes: Type.Array(MemoryScopeSchema),
    items: Type.Array(MemorySearchItemSchema),
  },
  { additionalProperties: false },
);

const DocumentsSortSchema = Type.Union([Type.Literal("recent"), Type.Literal("path")]);

export const WorkspaceDocumentRootSchema = Type.Object(
  {
    id: NonEmptyString,
    agentId: NonEmptyString,
    agentIds: Type.Array(NonEmptyString, { minItems: 1 }),
    label: NonEmptyString,
    path: NonEmptyString,
    fileCount: Type.Integer({ minimum: 0 }),
  },
  { additionalProperties: false },
);

export const WorkspaceDocumentEntrySchema = Type.Object(
  {
    rootId: NonEmptyString,
    agentId: NonEmptyString,
    agentIds: Type.Array(NonEmptyString, { minItems: 1 }),
    relativePath: NonEmptyString,
    name: NonEmptyString,
    category: Type.String(),
    extension: Type.String(),
    size: Type.Integer({ minimum: 0 }),
    mtimeMs: Type.Integer({ minimum: 0 }),
    preview: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export const DocumentsListParamsSchema = Type.Object(
  {
    agentId: Type.Optional(NonEmptyString),
    rootId: Type.Optional(NonEmptyString),
    query: Type.Optional(Type.String()),
    limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 1000 })),
    sort: Type.Optional(DocumentsSortSchema),
  },
  { additionalProperties: false },
);

export const DocumentsReadParamsSchema = Type.Object(
  {
    rootId: NonEmptyString,
    relativePath: NonEmptyString,
    maxBytes: Type.Optional(Type.Integer({ minimum: 1, maximum: 500_000 })),
  },
  { additionalProperties: false },
);

export const DocumentsListResultSchema = Type.Object(
  {
    updatedAt: Type.Integer({ minimum: 0 }),
    roots: Type.Array(WorkspaceDocumentRootSchema),
    items: Type.Array(WorkspaceDocumentEntrySchema),
  },
  { additionalProperties: false },
);

export const DocumentsReadResultSchema = Type.Object(
  {
    rootId: NonEmptyString,
    agentId: NonEmptyString,
    agentIds: Type.Array(NonEmptyString, { minItems: 1 }),
    relativePath: NonEmptyString,
    name: NonEmptyString,
    size: Type.Integer({ minimum: 0 }),
    mtimeMs: Type.Integer({ minimum: 0 }),
    content: Type.String(),
    truncated: Type.Boolean(),
  },
  { additionalProperties: false },
);

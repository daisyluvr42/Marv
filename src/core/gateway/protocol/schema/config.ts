import { Type } from "@sinclair/typebox";
import { NonEmptyString } from "./primitives.js";

export const ConfigGetParamsSchema = Type.Object({}, { additionalProperties: false });

export const ConfigSetParamsSchema = Type.Object(
  {
    raw: NonEmptyString,
    baseHash: Type.Optional(NonEmptyString),
  },
  { additionalProperties: false },
);

const ConfigApplyLikeParamsSchema = Type.Object(
  {
    raw: NonEmptyString,
    baseHash: Type.Optional(NonEmptyString),
    sessionKey: Type.Optional(Type.String()),
    note: Type.Optional(Type.String()),
    restartDelayMs: Type.Optional(Type.Integer({ minimum: 0 })),
  },
  { additionalProperties: false },
);

export const ConfigApplyParamsSchema = ConfigApplyLikeParamsSchema;
export const ConfigPatchParamsSchema = ConfigApplyLikeParamsSchema;

export const ConfigSchemaParamsSchema = Type.Object({}, { additionalProperties: false });

const ConfigPatchWriteMetaSchema = {
  actorId: Type.Optional(Type.String()),
  sessionKey: Type.Optional(Type.String()),
  note: Type.Optional(Type.String()),
  restartDelayMs: Type.Optional(Type.Integer({ minimum: 0 })),
} as const;

export const ConfigPatchesProposeParamsSchema = Type.Object(
  {
    naturalLanguage: NonEmptyString,
    scopeType: Type.Optional(Type.String()),
    scopeId: Type.Optional(Type.String()),
    autoCommit: Type.Optional(Type.Boolean()),
    ...ConfigPatchWriteMetaSchema,
  },
  { additionalProperties: false },
);

export const ConfigPatchesCommitParamsSchema = Type.Object(
  {
    proposalId: NonEmptyString,
    ...ConfigPatchWriteMetaSchema,
  },
  { additionalProperties: false },
);

export const ConfigRevisionsRollbackParamsSchema = Type.Object(
  {
    revision: NonEmptyString,
    ...ConfigPatchWriteMetaSchema,
  },
  { additionalProperties: false },
);

export const ConfigRevisionsListParamsSchema = Type.Object(
  {
    scopeType: Type.Optional(Type.String()),
    scopeId: Type.Optional(Type.String()),
    limit: Type.Optional(Type.Integer({ minimum: 1 })),
  },
  { additionalProperties: false },
);

export const UpdateRunParamsSchema = Type.Object(
  {
    sessionKey: Type.Optional(Type.String()),
    note: Type.Optional(Type.String()),
    restartDelayMs: Type.Optional(Type.Integer({ minimum: 0 })),
    timeoutMs: Type.Optional(Type.Integer({ minimum: 1 })),
  },
  { additionalProperties: false },
);

export const UpdateStatusParamsSchema = Type.Object(
  {
    timeoutMs: Type.Optional(Type.Integer({ minimum: 1 })),
  },
  { additionalProperties: false },
);

export const UpdateRollbackParamsSchema = Type.Object(
  {
    sessionKey: Type.Optional(Type.String()),
    note: Type.Optional(Type.String()),
    restartDelayMs: Type.Optional(Type.Integer({ minimum: 0 })),
    timeoutMs: Type.Optional(Type.Integer({ minimum: 1 })),
  },
  { additionalProperties: false },
);

export const ConfigUiHintSchema = Type.Object(
  {
    label: Type.Optional(Type.String()),
    help: Type.Optional(Type.String()),
    group: Type.Optional(Type.String()),
    order: Type.Optional(Type.Integer()),
    advanced: Type.Optional(Type.Boolean()),
    sensitive: Type.Optional(Type.Boolean()),
    placeholder: Type.Optional(Type.String()),
    itemTemplate: Type.Optional(Type.Unknown()),
  },
  { additionalProperties: false },
);

export const ConfigSchemaResponseSchema = Type.Object(
  {
    schema: Type.Unknown(),
    uiHints: Type.Record(Type.String(), ConfigUiHintSchema),
    version: NonEmptyString,
    generatedAt: NonEmptyString,
  },
  { additionalProperties: false },
);

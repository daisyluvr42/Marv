import { Type } from "@sinclair/typebox";
import { NonEmptyString } from "./primitives.js";

export const LedgerEventsQueryParamsSchema = Type.Object(
  {
    conversationId: NonEmptyString,
    taskId: Type.Optional(Type.String()),
    type: Type.Optional(Type.String()),
    fromTs: Type.Optional(Type.Integer({ minimum: 0 })),
    toTs: Type.Optional(Type.Integer({ minimum: 0 })),
    limit: Type.Optional(Type.Integer({ minimum: 1 })),
  },
  { additionalProperties: false },
);

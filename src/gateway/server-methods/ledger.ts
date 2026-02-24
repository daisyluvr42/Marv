import { queryLedgerEvents } from "../../ledger/event-store.js";
import { ErrorCodes, errorShape, validateLedgerEventsQueryParams } from "../protocol/index.js";
import type { GatewayRequestHandlers } from "./types.js";
import { assertValidParams } from "./validation.js";

export const ledgerHandlers: GatewayRequestHandlers = {
  "ledger.events.query": ({ params, respond }) => {
    if (
      !assertValidParams(params, validateLedgerEventsQueryParams, "ledger.events.query", respond)
    ) {
      return;
    }

    const rawConversationId = (params as { conversationId?: unknown }).conversationId;
    const conversationId = typeof rawConversationId === "string" ? rawConversationId.trim() : "";
    if (!conversationId) {
      respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "conversationId required"));
      return;
    }

    const events = queryLedgerEvents({
      conversationId,
      taskId: readOptionalString(params, "taskId"),
      type: readOptionalString(params, "type"),
      fromTs: readOptionalInteger(params, "fromTs"),
      toTs: readOptionalInteger(params, "toTs"),
      limit: readOptionalInteger(params, "limit"),
    });

    respond(
      true,
      {
        count: events.length,
        events: events.map((event) => ({
          id: event.id,
          eventId: event.eventId,
          taskId: event.taskId,
          conversationId: event.conversationId,
          type: event.type,
          ts: event.ts,
          actorId: event.actorId,
          payload: event.payload,
        })),
      },
      undefined,
    );
  },
};

function readOptionalString(params: Record<string, unknown>, key: string): string | undefined {
  const value = params[key];
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed || undefined;
}

function readOptionalInteger(params: Record<string, unknown>, key: string): number | undefined {
  const value = params[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  return Math.floor(value);
}

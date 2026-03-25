import { finalizeInboundContext } from "../inbound/context.js";
import type { FinalizedTurnContext, TurnContext } from "../support/templating.js";

export function buildTestCtx(overrides: Partial<TurnContext> = {}): FinalizedTurnContext {
  return finalizeInboundContext({
    Body: "",
    CommandBody: "",
    CommandSource: "text",
    From: "whatsapp:+1000",
    To: "whatsapp:+2000",
    ChatType: "direct",
    Provider: "whatsapp",
    Surface: "whatsapp",
    CommandAuthorized: false,
    ...overrides,
  });
}

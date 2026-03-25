import { expect } from "vitest";
import type { TurnContext } from "../../src/auto-reply/support/templating.js";
import { normalizeChatType } from "../../src/channels/chat-type.js";
import { resolveConversationLabel } from "../../src/channels/conversation-label.js";
import { validateSenderIdentity } from "../../src/channels/sender-identity.js";

export function expectInboundContextContract(ctx: TurnContext) {
  expect(validateSenderIdentity(ctx)).toEqual([]);

  expect(ctx.Body).toBeTypeOf("string");
  expect(ctx.BodyForAgent).toBeTypeOf("string");
  expect(ctx.BodyForCommands).toBeTypeOf("string");

  const chatType = normalizeChatType(ctx.ChatType);
  if (chatType && chatType !== "direct") {
    const label = ctx.ConversationLabel?.trim() || resolveConversationLabel(ctx);
    expect(label).toBeTruthy();
  }
}

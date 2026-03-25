import type { MarvConfig } from "../../core/config/config.js";
import type { DispatchInboundResult } from "../inbound/dispatch.js";
import {
  dispatchInboundMessageWithBufferedDispatcher,
  dispatchInboundMessageWithDispatcher,
} from "../inbound/dispatch.js";
import type { FinalizedTurnContext, TurnContext } from "../support/templating.js";
import type { GetReplyOptions } from "../support/types.js";
import type { ReplyDispatcherOptions, ReplyDispatcherWithTypingOptions } from "./dispatcher.js";

export async function dispatchReplyWithBufferedBlockDispatcher(params: {
  ctx: TurnContext | FinalizedTurnContext;
  cfg: MarvConfig;
  dispatcherOptions: ReplyDispatcherWithTypingOptions;
  replyOptions?: Omit<GetReplyOptions, "onToolResult" | "onBlockReply">;
  replyResolver?: typeof import("../index.js").getReplyFromConfig;
}): Promise<DispatchInboundResult> {
  return await dispatchInboundMessageWithBufferedDispatcher({
    ctx: params.ctx,
    cfg: params.cfg,
    dispatcherOptions: params.dispatcherOptions,
    replyResolver: params.replyResolver,
    replyOptions: params.replyOptions,
  });
}

export async function dispatchReplyWithDispatcher(params: {
  ctx: TurnContext | FinalizedTurnContext;
  cfg: MarvConfig;
  dispatcherOptions: ReplyDispatcherOptions;
  replyOptions?: Omit<GetReplyOptions, "onToolResult" | "onBlockReply">;
  replyResolver?: typeof import("../index.js").getReplyFromConfig;
}): Promise<DispatchInboundResult> {
  return await dispatchInboundMessageWithDispatcher({
    ctx: params.ctx,
    cfg: params.cfg,
    dispatcherOptions: params.dispatcherOptions,
    replyResolver: params.replyResolver,
    replyOptions: params.replyOptions,
  });
}

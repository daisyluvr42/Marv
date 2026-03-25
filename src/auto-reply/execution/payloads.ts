import type { ReplyToMode } from "../../core/config/types.js";
import { logVerbose } from "../../globals.js";
import { createBlockReplyPayloadKey, type BlockReplyPipeline } from "../delivery/block-pipeline.js";
import { normalizeReplyPayloadDirectives } from "../delivery/delivery.js";
import {
  applyReplyThreading,
  filterMessagingToolDuplicates,
  filterMessagingToolMediaDuplicates,
  isRenderablePayload,
  shouldSuppressMessagingToolReplies,
} from "../delivery/payloads.js";
import { stripHeartbeatToken } from "../support/heartbeat.js";
import type { OriginatingChannelType } from "../support/templating.js";
import { SILENT_REPLY_TOKEN } from "../support/tokens.js";
import {
  resolveReplyAckBehavior,
  type GetReplyOptions,
  type ReplyPayload,
} from "../support/types.js";
import { formatBunFetchSocketError, isBunFetchSocketError } from "./utils.js";

export function buildReplyPayloads(params: {
  payloads: ReplyPayload[];
  isHeartbeat: boolean;
  opts?: Pick<GetReplyOptions, "runMode" | "isHeartbeat">;
  didLogHeartbeatStrip: boolean;
  blockStreamingEnabled: boolean;
  blockReplyPipeline: BlockReplyPipeline | null;
  /** Payload keys sent directly (not via pipeline) during tool flush. */
  directlySentBlockKeys?: Set<string>;
  replyToMode: ReplyToMode;
  replyToChannel?: OriginatingChannelType;
  currentMessageId?: string;
  messageProvider?: string;
  messagingToolSentTexts?: string[];
  messagingToolSentMediaUrls?: string[];
  messagingToolSentTargets?: Parameters<
    typeof shouldSuppressMessagingToolReplies
  >[0]["messagingToolSentTargets"];
  originatingTo?: string;
  accountId?: string;
}): { replyPayloads: ReplyPayload[]; didLogHeartbeatStrip: boolean } {
  let didLogHeartbeatStrip = params.didLogHeartbeatStrip;
  const ackBehavior = resolveReplyAckBehavior(params.opts);
  const sanitizedPayloads = params.isHeartbeat
    ? params.payloads
    : params.payloads.flatMap((payload) => {
        let text = payload.text;
        const ackToken = ackBehavior?.token ?? "HEARTBEAT_OK";

        if (payload.isError && text && isBunFetchSocketError(text)) {
          text = formatBunFetchSocketError(text);
        }

        if (!text || !text.includes(ackToken)) {
          return [{ ...payload, text }];
        }
        const stripped = stripHeartbeatToken(text, {
          mode: "message",
          token: ackBehavior?.token,
          maxAckChars: ackBehavior?.maxAckChars,
        });
        if (stripped.didStrip && !didLogHeartbeatStrip) {
          didLogHeartbeatStrip = true;
          logVerbose("Stripped stray HEARTBEAT_OK token from reply");
        }
        const hasMedia = Boolean(payload.mediaUrl) || (payload.mediaUrls?.length ?? 0) > 0;
        if (stripped.shouldSkip && !hasMedia) {
          return [];
        }
        return [{ ...payload, text: stripped.text }];
      });

  const replyTaggedPayloads: ReplyPayload[] = applyReplyThreading({
    payloads: sanitizedPayloads,
    replyToMode: params.replyToMode,
    replyToChannel: params.replyToChannel,
    currentMessageId: params.currentMessageId,
  })
    .map(
      (payload) =>
        normalizeReplyPayloadDirectives({
          payload,
          currentMessageId: params.currentMessageId,
          silentToken: SILENT_REPLY_TOKEN,
          parseMode: "always",
        }).payload,
    )
    .filter(isRenderablePayload);

  // Drop final payloads only when block streaming succeeded end-to-end.
  // If streaming aborted (e.g., timeout), fall back to final payloads.
  const shouldDropFinalPayloads =
    params.blockStreamingEnabled &&
    Boolean(params.blockReplyPipeline?.didStream()) &&
    !params.blockReplyPipeline?.isAborted();
  const messagingToolSentTexts = params.messagingToolSentTexts ?? [];
  const messagingToolSentTargets = params.messagingToolSentTargets ?? [];
  const suppressMessagingToolReplies = shouldSuppressMessagingToolReplies({
    messageProvider: params.messageProvider,
    messagingToolSentTargets,
    originatingTo: params.originatingTo,
    accountId: params.accountId,
  });
  const dedupedPayloads = filterMessagingToolDuplicates({
    payloads: replyTaggedPayloads,
    sentTexts: messagingToolSentTexts,
  });
  const mediaFilteredPayloads = filterMessagingToolMediaDuplicates({
    payloads: dedupedPayloads,
    sentMediaUrls: params.messagingToolSentMediaUrls ?? [],
  });
  // Filter out payloads already sent via pipeline or directly during tool flush.
  const filteredPayloads = shouldDropFinalPayloads
    ? []
    : params.blockStreamingEnabled
      ? mediaFilteredPayloads.filter(
          (payload) => !params.blockReplyPipeline?.hasSentPayload(payload),
        )
      : params.directlySentBlockKeys?.size
        ? mediaFilteredPayloads.filter(
            (payload) => !params.directlySentBlockKeys!.has(createBlockReplyPayloadKey(payload)),
          )
        : mediaFilteredPayloads;
  const replyPayloads = suppressMessagingToolReplies ? [] : filteredPayloads;

  return {
    replyPayloads,
    didLogHeartbeatStrip,
  };
}

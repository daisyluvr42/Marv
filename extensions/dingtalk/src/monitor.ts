import { createReplyPrefixOptions, type MarvConfig, type RuntimeEnv } from "marv/plugin-sdk";
import { resolveDingTalkAccount } from "./accounts.js";
import { sendDingTalkReply } from "./api.js";
import { getDingTalkRuntime } from "./runtime.js";
import type {
  DingTalkInboundMessage,
  DingTalkReplyContext,
  ResolvedDingTalkAccount,
} from "./types.js";

type MonitorOptions = {
  cfg: MarvConfig;
  account: ResolvedDingTalkAccount;
  runtime: RuntimeEnv;
  abortSignal: AbortSignal;
  statusSink?: (patch: Record<string, unknown>) => void;
};

function extractInboundText(message: DingTalkInboundMessage): string {
  if (message.msgtype === "text") {
    return message.text?.content?.trim() ?? "";
  }
  if (message.msgtype === "richText") {
    return (message.content?.richText ?? [])
      .map((entry) => entry.text ?? "")
      .join("")
      .trim();
  }
  if (message.msgtype === "audio") {
    return message.content?.recognition?.trim() ?? "";
  }
  return message.text?.content?.trim() ?? "";
}

function isAllowedSender(senderId: string, allowFrom: Array<string | number> | undefined): boolean {
  const normalizedSender = senderId.trim().toLowerCase();
  const entries = (allowFrom ?? []).map((entry) => String(entry).trim().toLowerCase());
  if (entries.length === 0) {
    return false;
  }
  if (entries.includes("*")) {
    return true;
  }
  return entries.includes(normalizedSender);
}

async function processInboundMessage(params: {
  cfg: MarvConfig;
  account: ResolvedDingTalkAccount;
  message: DingTalkInboundMessage;
  runtime: RuntimeEnv;
  statusSink?: (patch: Record<string, unknown>) => void;
}) {
  const { cfg, account, message, runtime, statusSink } = params;
  const core = getDingTalkRuntime();
  const bodyForAgent = extractInboundText(message);
  const conversationId = message.conversationId?.trim() ?? "";
  const senderId = (message.senderStaffId?.trim() || message.senderId?.trim() || "").trim();
  const sessionWebhook = message.sessionWebhook?.trim() ?? "";
  if (!bodyForAgent || !conversationId || !senderId || !sessionWebhook) {
    return;
  }

  const isGroup = message.conversationType === "2";
  if (!isGroup) {
    const policy = account.config.dmPolicy ?? "allowlist";
    if (policy === "disabled") {
      return;
    }
    if (policy === "allowlist" && !isAllowedSender(senderId, account.config.allowFrom)) {
      return;
    }
  } else {
    const policy = account.config.groupPolicy ?? "disabled";
    if (policy === "disabled") {
      return;
    }
    if (policy === "allowlist" && !isAllowedSender(senderId, account.config.groupAllowFrom)) {
      return;
    }
  }

  const route = core.channel.routing.resolveAgentRoute({
    cfg,
    channel: "dingtalk",
    accountId: account.accountId,
    peer: {
      kind: isGroup ? "group" : "direct",
      id: conversationId,
    },
  });
  const fromLabel = isGroup ? `group:${conversationId}` : message.senderNick?.trim() || senderId;
  const storePath = core.channel.session.resolveStorePath(cfg.session?.store, {
    agentId: route.agentId,
  });
  const envelopeOptions = core.channel.reply.resolveEnvelopeFormatOptions(cfg);
  const previousTimestamp = core.channel.session.readSessionUpdatedAt({
    storePath,
    sessionKey: route.sessionKey,
  });
  const body = core.channel.reply.formatAgentEnvelope({
    channel: "DingTalk",
    from: fromLabel,
    timestamp: typeof message.createAt === "number" ? message.createAt : undefined,
    previousTimestamp,
    envelope: envelopeOptions,
    body: bodyForAgent,
  });
  const ctxPayload = core.channel.reply.finalizeInboundContext({
    Body: body,
    BodyForAgent: bodyForAgent,
    RawBody: bodyForAgent,
    CommandBody: bodyForAgent,
    From: `dingtalk:${senderId}`,
    To: `dingtalk:${conversationId}`,
    SessionKey: route.sessionKey,
    AccountId: route.accountId,
    ChatType: isGroup ? "channel" : "direct",
    ConversationLabel: fromLabel,
    SenderName: message.senderNick?.trim() || undefined,
    SenderId: senderId,
    Provider: "dingtalk",
    Surface: "dingtalk",
    MessageSid: message.msgId,
    MessageSidFull: message.msgId,
    OriginatingChannel: "dingtalk",
    OriginatingTo: `dingtalk:${conversationId}`,
  });

  void core.channel.session
    .recordSessionMetaFromInbound({
      storePath,
      sessionKey: ctxPayload.SessionKey ?? route.sessionKey,
      ctx: ctxPayload,
    })
    .catch((error) => runtime.error?.(`dingtalk: failed updating session meta: ${String(error)}`));

  const replyContext: DingTalkReplyContext = {
    sessionWebhook,
    conversationType: message.conversationType === "2" ? "2" : "1",
    userId: senderId,
    openConversationId: conversationId,
  };
  const { onModelSelected, ...prefixOptions } = createReplyPrefixOptions({
    cfg,
    agentId: route.agentId,
    channel: "dingtalk",
    accountId: route.accountId,
  });

  await core.channel.reply.dispatchReplyWithBufferedBlockDispatcher({
    ctx: ctxPayload,
    cfg,
    dispatcherOptions: {
      ...prefixOptions,
      deliver: async (payload) => {
        const mediaLinks = payload.mediaUrls?.length
          ? `\n${payload.mediaUrls.join("\n")}`
          : payload.mediaUrl
            ? `\n${payload.mediaUrl}`
            : "";
        await sendDingTalkReply({
          account,
          context: replyContext,
          text: `${payload.text ?? ""}${mediaLinks}`.trim(),
        });
        statusSink?.({ lastOutboundAt: Date.now() });
      },
      onError: (error, info) => {
        runtime.error?.(`dingtalk ${info.kind} reply failed: ${String(error)}`);
      },
    },
    replyOptions: {
      onModelSelected,
    },
  });
}

export async function monitorDingTalkProvider(
  opts: MonitorOptions,
): Promise<{ stop: () => Promise<void> }> {
  const { account, abortSignal, runtime, cfg, statusSink } = opts;
  const { DWClient, TOPIC_ROBOT } = await import("dingtalk-stream");
  const client = new DWClient({
    clientId: account.clientId,
    clientSecret: account.clientSecret,
    keepAlive: true,
  });

  client.registerCallbackListener(
    TOPIC_ROBOT,
    async (event: { headers?: { messageId?: string }; data?: string }) => {
      try {
        const messageId = event.headers?.messageId;
        if (messageId && typeof client.socketCallBackResponse === "function") {
          client.socketCallBackResponse(messageId, { success: true });
        }
        const raw = typeof event.data === "string" ? JSON.parse(event.data) : {};
        statusSink?.({ lastInboundAt: Date.now() });
        await processInboundMessage({
          cfg,
          account,
          message: raw as DingTalkInboundMessage,
          runtime,
          statusSink,
        });
      } catch (error) {
        runtime.error?.(`dingtalk inbound failed: ${String(error)}`);
      }
    },
  );

  await client.connect();

  const stop = async () => {
    if (typeof client.disconnect === "function") {
      await client.disconnect();
    }
  };

  if (abortSignal.aborted) {
    await stop();
    return { stop };
  }
  abortSignal.addEventListener(
    "abort",
    () => {
      void stop();
    },
    { once: true },
  );

  return { stop };
}

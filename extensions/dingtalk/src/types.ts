import type { BaseProbeResult } from "marv/plugin-sdk";

export type DingTalkDmPolicy = "open" | "allowlist" | "disabled";
export type DingTalkGroupPolicy = "open" | "allowlist" | "disabled";

export type DingTalkAccountConfig = {
  enabled?: boolean;
  name?: string;
  clientId?: string;
  clientSecret?: string;
  robotCode?: string;
  dmPolicy?: DingTalkDmPolicy;
  allowFrom?: Array<string | number>;
  groupPolicy?: DingTalkGroupPolicy;
  groupAllowFrom?: Array<string | number>;
};

export type DingTalkConfig = DingTalkAccountConfig & {
  accounts?: Record<string, DingTalkAccountConfig>;
};

export type ResolvedDingTalkAccount = {
  accountId: string;
  enabled: boolean;
  configured: boolean;
  name?: string;
  clientId?: string;
  clientSecret?: string;
  robotCode?: string;
  config: DingTalkAccountConfig;
};

export type DingTalkProbeResult = BaseProbeResult<string> & {
  accountId?: string;
  clientId?: string;
  robotCode?: string;
  stage?: "credentials" | "token";
};

export type DingTalkTarget = { kind: "user"; value: string } | { kind: "group"; value: string };

export type DingTalkInboundMessage = {
  msgId?: string;
  msgtype?: string;
  createAt?: number;
  text?: { content?: string };
  content?: {
    recognition?: string;
    richText?: Array<{ text?: string }>;
  };
  conversationType?: "1" | "2";
  conversationId?: string;
  senderId?: string;
  senderStaffId?: string;
  senderNick?: string;
  chatbotUserId?: string;
  sessionWebhook?: string;
};

export type DingTalkReplyContext = {
  sessionWebhook: string;
  conversationType: "1" | "2";
  userId?: string;
  openConversationId?: string;
};

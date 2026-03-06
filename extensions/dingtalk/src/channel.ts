import {
  buildBaseChannelStatusSummary,
  buildChannelConfigSchema,
  createDefaultChannelRuntimeState,
  DEFAULT_ACCOUNT_ID,
  deleteAccountFromConfigSection,
  setAccountEnabledInConfigSection,
  type ChannelMeta,
  type ChannelPlugin,
  type ChannelStatusIssue,
} from "agentmarv/plugin-sdk";
import {
  resolveDingTalkAccount,
  listDingTalkAccountIds,
  resolveDefaultDingTalkAccountId,
} from "./accounts.js";
import { probeDingTalk, sendDingTalkText } from "./api.js";
import { DingTalkConfigSchema } from "./config-schema.js";
import { looksLikeDingTalkId, normalizeDingTalkTarget } from "./targets.js";
import type { DingTalkProbeResult, ResolvedDingTalkAccount } from "./types.js";

const meta: ChannelMeta = {
  id: "dingtalk",
  label: "DingTalk",
  selectionLabel: "DingTalk (钉钉)",
  docsPath: "/channels/dingtalk",
  docsLabel: "dingtalk",
  blurb: "DingTalk enterprise messaging.",
  order: 75,
};

export const dingtalkPlugin: ChannelPlugin<ResolvedDingTalkAccount, DingTalkProbeResult> = {
  id: "dingtalk",
  meta,
  capabilities: {
    chatTypes: ["direct", "group"],
    media: false,
    reactions: false,
    edit: false,
    reply: true,
  },
  reload: { configPrefixes: ["channels.dingtalk"] },
  configSchema: buildChannelConfigSchema(DingTalkConfigSchema),
  config: {
    listAccountIds: (cfg) => listDingTalkAccountIds(cfg),
    resolveAccount: (cfg, accountId) => resolveDingTalkAccount({ cfg, accountId }),
    defaultAccountId: (cfg) => resolveDefaultDingTalkAccountId(cfg),
    setAccountEnabled: ({ cfg, accountId, enabled }) =>
      setAccountEnabledInConfigSection({
        cfg,
        sectionKey: "dingtalk",
        accountId,
        enabled,
        allowTopLevel: true,
      }),
    deleteAccount: ({ cfg, accountId }) =>
      deleteAccountFromConfigSection({
        cfg,
        sectionKey: "dingtalk",
        accountId,
        clearBaseFields: [
          "clientId",
          "clientSecret",
          "robotCode",
          "dmPolicy",
          "allowFrom",
          "groupPolicy",
          "groupAllowFrom",
        ],
      }),
    isConfigured: (account) => account.configured,
    describeAccount: (account) => ({
      accountId: account.accountId,
      enabled: account.enabled,
      configured: account.configured,
      name: account.name,
      clientId: account.clientId,
      robotCode: account.robotCode,
    }),
    resolveAllowFrom: ({ cfg, accountId }) =>
      (resolveDingTalkAccount({ cfg, accountId }).config.allowFrom ?? []).map((entry) =>
        String(entry),
      ),
    formatAllowFrom: ({ allowFrom }) =>
      allowFrom.map((entry) => String(entry).trim().toLowerCase()).filter(Boolean),
  },
  outbound: {
    deliveryMode: "direct",
    resolveTarget: ({ to }) => {
      const normalized = normalizeDingTalkTarget(to?.trim() ?? "");
      if (!normalized) {
        return {
          ok: false,
          error: new Error('invalid DingTalk target (use "user:<id>" or "group:<conversationId>")'),
        };
      }
      return { ok: true, to: normalized };
    },
    sendText: async ({ cfg, to, text, accountId }) => {
      const account = resolveDingTalkAccount({ cfg, accountId });
      await sendDingTalkText({
        account,
        target: to,
        text,
      });
      return {
        channel: "dingtalk",
        messageId: "",
        chatId: to,
      };
    },
  },
  messaging: {
    normalizeTarget: (raw) => normalizeDingTalkTarget(raw) ?? undefined,
    targetResolver: {
      looksLikeId: looksLikeDingTalkId,
      hint: "<user:id|group:conversationId>",
    },
  },
  status: {
    defaultRuntime: createDefaultChannelRuntimeState(DEFAULT_ACCOUNT_ID, {
      lastInboundAt: null,
      lastOutboundAt: null,
    }),
    buildChannelSummary: ({ snapshot }) => ({
      ...buildBaseChannelStatusSummary(snapshot),
      lastInboundAt: snapshot.lastInboundAt ?? null,
      lastOutboundAt: snapshot.lastOutboundAt ?? null,
      probe: snapshot.probe,
      lastProbeAt: snapshot.lastProbeAt ?? null,
    }),
    probeAccount: async ({ account }) => await probeDingTalk(account),
    buildAccountSnapshot: ({ account, runtime, probe }) => ({
      accountId: account.accountId,
      enabled: account.enabled,
      configured: account.configured,
      name: account.name,
      clientId: account.clientId,
      robotCode: account.robotCode,
      running: runtime?.running ?? false,
      lastStartAt: runtime?.lastStartAt ?? null,
      lastStopAt: runtime?.lastStopAt ?? null,
      lastError: runtime?.lastError ?? null,
      lastInboundAt: runtime?.lastInboundAt ?? null,
      lastOutboundAt: runtime?.lastOutboundAt ?? null,
      probe,
    }),
    collectStatusIssues: (accounts) =>
      accounts.flatMap((entry) => {
        const issues: ChannelStatusIssue[] = [];
        const probe = entry.probe as DingTalkProbeResult | undefined;
        if (entry.enabled !== false && entry.configured === true && probe?.ok === false) {
          issues.push({
            channel: "dingtalk",
            accountId: String(entry.accountId ?? DEFAULT_ACCOUNT_ID),
            kind: "runtime",
            message: `DingTalk probe failed: ${probe.error ?? "unknown error"}`,
          });
        }
        return issues;
      }),
  },
  gateway: {
    startAccount: async (ctx) => {
      const { monitorDingTalkProvider } = await import("./monitor.js");
      return await monitorDingTalkProvider({
        cfg: ctx.cfg,
        account: ctx.account,
        runtime: ctx.runtime,
        abortSignal: ctx.abortSignal,
        statusSink: (patch) => ctx.setStatus({ accountId: ctx.accountId, ...patch }),
      });
    },
  },
};

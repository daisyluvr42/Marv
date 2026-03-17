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
  resolveWeChatAccount,
  listWeChatAccountIds,
  resolveDefaultWeChatAccountId,
} from "./accounts.js";
import { probeWeChat, sendWeChatText } from "./api.js";
import { WeChatConfigSchema } from "./config-schema.js";
import type { WeChatProbeResult, ResolvedWeChatAccount } from "./types.js";

const meta: ChannelMeta = {
  id: "wechat",
  label: "WeChat",
  selectionLabel: "WeChat (微信)",
  docsPath: "/channels/wechat",
  docsLabel: "wechat",
  blurb: "WeChat messaging via Wechaty puppet providers.",
  order: 76,
};

export const wechatPlugin: ChannelPlugin<ResolvedWeChatAccount, WeChatProbeResult> = {
  id: "wechat",
  meta,
  capabilities: {
    chatTypes: ["direct", "group"],
    media: false,
    reactions: false,
    edit: false,
    reply: true,
  },
  reload: { configPrefixes: ["channels.wechat"] },
  configSchema: buildChannelConfigSchema(WeChatConfigSchema),
  config: {
    listAccountIds: (cfg) => listWeChatAccountIds(cfg),
    resolveAccount: (cfg, accountId) => resolveWeChatAccount({ cfg, accountId }),
    defaultAccountId: (cfg) => resolveDefaultWeChatAccountId(cfg),
    setAccountEnabled: ({ cfg, accountId, enabled }) =>
      setAccountEnabledInConfigSection({
        cfg,
        sectionKey: "wechat",
        accountId,
        enabled,
        allowTopLevel: true,
      }),
    deleteAccount: ({ cfg, accountId }) =>
      deleteAccountFromConfigSection({
        cfg,
        sectionKey: "wechat",
        accountId,
        clearBaseFields: [
          "puppet",
          "puppetToken",
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
      puppet: account.puppet,
    }),
    resolveAllowFrom: ({ cfg, accountId }) =>
      (resolveWeChatAccount({ cfg, accountId }).config.allowFrom ?? []).map((entry) =>
        String(entry),
      ),
    formatAllowFrom: ({ allowFrom }) =>
      allowFrom.map((entry) => String(entry).trim().toLowerCase()).filter(Boolean),
  },
  outbound: {
    deliveryMode: "direct",
    resolveTarget: ({ to }) => {
      const trimmed = to?.trim() ?? "";
      if (!trimmed) {
        return {
          ok: false,
          error: new Error("invalid WeChat target (provide a contact or room ID)"),
        };
      }
      return { ok: true, to: trimmed };
    },
    sendText: async ({ cfg, to, text, accountId }) => {
      const account = resolveWeChatAccount({ cfg, accountId });
      await sendWeChatText({ account, target: to, text });
      return {
        channel: "wechat",
        messageId: "",
        chatId: to,
      };
    },
  },
  messaging: {
    normalizeTarget: (raw) => raw?.trim() || undefined,
    targetResolver: {
      looksLikeId: (raw) => Boolean(raw?.trim()),
      hint: "<contactId|roomId>",
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
    probeAccount: async ({ account }) => await probeWeChat(account),
    buildAccountSnapshot: ({ account, runtime, probe }) => ({
      accountId: account.accountId,
      enabled: account.enabled,
      configured: account.configured,
      name: account.name,
      puppet: account.puppet,
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
        const probe = entry.probe as WeChatProbeResult | undefined;
        if (entry.enabled !== false && entry.configured === true && probe?.ok === false) {
          issues.push({
            channel: "wechat",
            accountId: String(entry.accountId ?? DEFAULT_ACCOUNT_ID),
            kind: "runtime",
            message: `WeChat probe failed: ${probe.error ?? "unknown error"}`,
          });
        }
        return issues;
      }),
  },
  gateway: {
    startAccount: async (ctx) => {
      const { monitorWeChatProvider } = await import("./monitor.js");
      return await monitorWeChatProvider({
        cfg: ctx.cfg,
        account: ctx.account,
        runtime: ctx.runtime,
        abortSignal: ctx.abortSignal,
        statusSink: (patch) => ctx.setStatus({ accountId: ctx.accountId, ...patch }),
      });
    },
  },
};

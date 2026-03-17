import { DEFAULT_ACCOUNT_ID } from "agentmarv/plugin-sdk";
import type { MarvConfig } from "agentmarv/plugin-sdk";
import type { ResolvedWeChatAccount, WeChatAccountConfig, WeChatConfig } from "./types.js";

function readWeChatConfig(cfg: MarvConfig): WeChatConfig | undefined {
  return cfg.channels?.wechat as WeChatConfig | undefined;
}

function mergeAccountConfig(
  base: WeChatAccountConfig,
  override?: WeChatAccountConfig,
): WeChatAccountConfig {
  if (!override) return { ...base };
  return {
    ...base,
    ...override,
    allowFrom: override.allowFrom ?? base.allowFrom,
    groupAllowFrom: override.groupAllowFrom ?? base.groupAllowFrom,
  };
}

export function listWeChatAccountIds(cfg: MarvConfig): string[] {
  const config = readWeChatConfig(cfg);
  const accountIds = Object.keys(config?.accounts ?? {});
  return [DEFAULT_ACCOUNT_ID, ...accountIds];
}

export function resolveDefaultWeChatAccountId(_cfg: MarvConfig): string {
  return DEFAULT_ACCOUNT_ID;
}

export function resolveWeChatAccount(params: {
  cfg: MarvConfig;
  accountId?: string | null;
}): ResolvedWeChatAccount {
  const config = readWeChatConfig(params.cfg);
  const accountId = params.accountId?.trim() || DEFAULT_ACCOUNT_ID;
  const base: WeChatAccountConfig = {
    enabled: config?.enabled,
    name: undefined,
    puppet: config?.puppet ?? "wechaty-puppet-wechat4u",
    puppetToken: config?.puppetToken,
    dmPolicy: config?.dmPolicy ?? "pairing",
    allowFrom: config?.allowFrom ?? [],
    groupPolicy: config?.groupPolicy ?? "disabled",
    groupAllowFrom: config?.groupAllowFrom ?? [],
    groups: config?.groups,
  };
  const named = accountId === DEFAULT_ACCOUNT_ID ? undefined : config?.accounts?.[accountId];
  const merged = mergeAccountConfig(base, named);
  const configured = Boolean(merged.puppet?.trim());
  return {
    accountId,
    enabled: merged.enabled !== false,
    configured,
    name: merged.name?.trim() || undefined,
    puppet: merged.puppet?.trim() || undefined,
    puppetToken: merged.puppetToken?.trim() || undefined,
    config: merged,
  };
}

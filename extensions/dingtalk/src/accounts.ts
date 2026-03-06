import { DEFAULT_ACCOUNT_ID } from "agentmarv/plugin-sdk";
import type { MarvConfig } from "agentmarv/plugin-sdk";
import type { DingTalkAccountConfig, DingTalkConfig, ResolvedDingTalkAccount } from "./types.js";

function readDingTalkConfig(cfg: MarvConfig): DingTalkConfig | undefined {
  return cfg.channels?.dingtalk as DingTalkConfig | undefined;
}

function mergeAccountConfig(base: DingTalkAccountConfig, override?: DingTalkAccountConfig) {
  if (!override) {
    return { ...base };
  }
  return {
    ...base,
    ...override,
    allowFrom: override.allowFrom ?? base.allowFrom,
    groupAllowFrom: override.groupAllowFrom ?? base.groupAllowFrom,
  };
}

export function listDingTalkAccountIds(cfg: MarvConfig): string[] {
  const config = readDingTalkConfig(cfg);
  const accountIds = Object.keys(config?.accounts ?? {});
  return [DEFAULT_ACCOUNT_ID, ...accountIds];
}

export function resolveDefaultDingTalkAccountId(_cfg: MarvConfig): string {
  return DEFAULT_ACCOUNT_ID;
}

export function resolveDingTalkAccount(params: {
  cfg: MarvConfig;
  accountId?: string | null;
}): ResolvedDingTalkAccount {
  const config = readDingTalkConfig(params.cfg);
  const accountId = params.accountId?.trim() || DEFAULT_ACCOUNT_ID;
  const base: DingTalkAccountConfig = {
    enabled: config?.enabled,
    name: undefined,
    clientId: config?.clientId,
    clientSecret: config?.clientSecret,
    robotCode: config?.robotCode,
    dmPolicy: config?.dmPolicy ?? "allowlist",
    allowFrom: config?.allowFrom ?? [],
    groupPolicy: config?.groupPolicy ?? "disabled",
    groupAllowFrom: config?.groupAllowFrom ?? [],
  };
  const named = accountId === DEFAULT_ACCOUNT_ID ? undefined : config?.accounts?.[accountId];
  const merged = mergeAccountConfig(base, named);
  const configured = Boolean(merged.clientId?.trim() && merged.clientSecret?.trim());
  return {
    accountId,
    enabled: merged.enabled !== false,
    configured,
    name: merged.name?.trim() || undefined,
    clientId: merged.clientId?.trim() || undefined,
    clientSecret: merged.clientSecret?.trim() || undefined,
    robotCode: merged.robotCode?.trim() || undefined,
    config: merged,
  };
}

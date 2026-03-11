import { getChannelPlugin, listChannelPlugins } from "../channels/plugins/index.js";
import type { ChannelId } from "../channels/plugins/types.js";
import type { MarvConfig } from "../core/config/config.js";
import { DEFAULT_ACCOUNT_ID } from "../routing/session-key.js";

type ProviderAccountStatus = {
  provider: ChannelId;
  accountId: string;
  name?: string;
  state: "linked" | "not linked" | "configured" | "not configured" | "enabled" | "disabled";
  enabled?: boolean;
  configured?: boolean;
};

function providerAccountKey(provider: ChannelId, accountId?: string) {
  return `${provider}:${accountId ?? DEFAULT_ACCOUNT_ID}`;
}

function formatChannelAccountLabel(params: {
  provider: ChannelId;
  accountId: string;
  name?: string;
}): string {
  const label = getChannelPlugin(params.provider)?.meta.label ?? params.provider;
  const account = params.name?.trim()
    ? `${params.accountId} (${params.name.trim()})`
    : params.accountId;
  return `${label} ${account}`;
}

function formatProviderState(entry: ProviderAccountStatus): string {
  const parts = [entry.state];
  if (entry.enabled === false && entry.state !== "disabled") {
    parts.push("disabled");
  }
  return parts.join(", ");
}

export async function buildProviderStatusIndex(
  cfg: MarvConfig,
): Promise<Map<string, ProviderAccountStatus>> {
  const map = new Map<string, ProviderAccountStatus>();

  for (const plugin of listChannelPlugins()) {
    const accountIds = plugin.config.listAccountIds(cfg);
    for (const accountId of accountIds) {
      const account = plugin.config.resolveAccount(cfg, accountId);
      const snapshot = plugin.config.describeAccount?.(account, cfg);
      const enabled = plugin.config.isEnabled
        ? plugin.config.isEnabled(account, cfg)
        : typeof snapshot?.enabled === "boolean"
          ? snapshot.enabled
          : (account as { enabled?: boolean }).enabled;
      const configured = plugin.config.isConfigured
        ? await plugin.config.isConfigured(account, cfg)
        : snapshot?.configured;
      const resolvedEnabled = typeof enabled === "boolean" ? enabled : true;
      const resolvedConfigured = typeof configured === "boolean" ? configured : true;
      const state =
        plugin.status?.resolveAccountState?.({
          account,
          cfg,
          configured: resolvedConfigured,
          enabled: resolvedEnabled,
        }) ??
        (typeof snapshot?.linked === "boolean"
          ? snapshot.linked
            ? "linked"
            : "not linked"
          : resolvedConfigured
            ? "configured"
            : "not configured");
      const name = snapshot?.name ?? (account as { name?: string }).name;
      map.set(providerAccountKey(plugin.id, accountId), {
        provider: plugin.id,
        accountId,
        name,
        state,
        enabled,
        configured,
      });
    }
  }

  return map;
}

function shouldShowProviderEntry(entry: ProviderAccountStatus, cfg: MarvConfig): boolean {
  const plugin = getChannelPlugin(entry.provider);
  if (!plugin) {
    return Boolean(entry.configured);
  }
  if (plugin.meta.showConfigured === false) {
    const providerConfig = (cfg as Record<string, unknown>)[plugin.id];
    return Boolean(entry.configured) || Boolean(providerConfig);
  }
  return Boolean(entry.configured);
}

function formatProviderEntry(entry: ProviderAccountStatus): string {
  const label = formatChannelAccountLabel({
    provider: entry.provider,
    accountId: entry.accountId,
    name: entry.name,
  });
  return `${label}: ${formatProviderState(entry)}`;
}

export function listProvidersForAgent(params: {
  summaryIsDefault: boolean;
  cfg: MarvConfig;
  providerStatus: Map<string, ProviderAccountStatus>;
}): string[] {
  if (!params.summaryIsDefault) {
    return [];
  }

  const providerLines: string[] = [];
  for (const entry of params.providerStatus.values()) {
    if (shouldShowProviderEntry(entry, params.cfg)) {
      providerLines.push(formatProviderEntry(entry));
    }
  }
  return providerLines;
}

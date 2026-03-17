export type WeChatAccountConfig = {
  enabled?: boolean;
  /** Wechaty puppet provider name (e.g. "wechaty-puppet-padlocal", "wechaty-puppet-xp"). */
  puppet?: string;
  /** Puppet-specific token (required by some puppets). */
  puppetToken?: string;
  /** Name label for this account. */
  name?: string;
  /** DM policy: "pairing" | "allowlist" | "open" | "disabled". */
  dmPolicy?: string;
  /** Sender allowlist (WeChat contact IDs). */
  allowFrom?: Array<string | number>;
  /** Group policy: "open" | "disabled" | "allowlist". */
  groupPolicy?: string;
  /** Group-specific sender allowlist. */
  groupAllowFrom?: Array<string | number>;
  /** Per-group overrides. */
  groups?: Record<string, { requireMention?: boolean }>;
};

export type WeChatConfig = WeChatAccountConfig & {
  accounts?: Record<string, WeChatAccountConfig>;
};

export type ResolvedWeChatAccount = {
  accountId: string;
  enabled: boolean;
  configured: boolean;
  name?: string;
  puppet?: string;
  puppetToken?: string;
  config: WeChatAccountConfig;
};

export type WeChatProbeResult = {
  ok: boolean;
  loggedIn?: boolean;
  userName?: string;
  error?: string;
};

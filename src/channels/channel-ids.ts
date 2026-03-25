// Leaf module: channel ID constants and type.
// Extracted from registry.ts to break circular dependencies
// (channels/registry ↔ channels/plugins/types.core, logging/subsystem, etc.).

// Channel docking: add new core channels here (order + meta + aliases), then
// register the plugin in its extension entrypoint and keep protocol IDs in sync.
export const CHAT_CHANNEL_ORDER = [
  "telegram",
  "whatsapp",
  "discord",
  "irc",
  "googlechat",
  "slack",
  "signal",
  "imessage",
] as const;

export type ChatChannelId = (typeof CHAT_CHANNEL_ORDER)[number];

/** Broader channel ID type: any built-in channel or an extension-provided string. */
export type ChannelId = ChatChannelId | (string & {});

export const CHANNEL_IDS = [...CHAT_CHANNEL_ORDER] as const;

export const DEFAULT_CHAT_CHANNEL: ChatChannelId = "whatsapp";

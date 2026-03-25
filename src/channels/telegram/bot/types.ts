import type { Message, UserFromGetMe } from "@grammyjs/types";
import type { MarvConfig, ReplyToMode } from "../../../core/config/config.js";
import type { RuntimeEnv } from "../../../runtime.js";

/** App-specific stream mode for Telegram stream previews. */
export type TelegramStreamMode = "off" | "partial" | "block";

/**
 * Minimal context projection from Grammy's Context class.
 * Decouples the message processing pipeline from Grammy's full Context,
 * and allows constructing synthetic contexts for debounced/combined messages.
 */
export type TelegramContext = {
  message: Message;
  me?: UserFromGetMe;
  getFile: () => Promise<{ file_path?: string }>;
};

export type TelegramBotOptions = {
  token: string;
  accountId?: string;
  runtime?: RuntimeEnv;
  requireMention?: boolean;
  allowFrom?: Array<string | number>;
  groupAllowFrom?: Array<string | number>;
  mediaMaxMb?: number;
  replyToMode?: ReplyToMode;
  proxyFetch?: typeof fetch;
  config?: MarvConfig;
  updateOffset?: {
    lastUpdateId?: number | null;
    onUpdateId?: (updateId: number) => void | Promise<void>;
  };
  testTimings?: {
    mediaGroupFlushMs?: number;
    textFragmentGapMs?: number;
  };
};

export type { StickerMetadata } from "./sticker-types.js";

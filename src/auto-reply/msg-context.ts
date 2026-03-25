import type { ChannelId } from "../channels/channel-ids.js";
import type { StickerMetadata } from "../channels/telegram/bot/sticker-types.js";
import type { PromptMediaRef } from "../contracts/media-ref.js";
import type { MediaPromptCompatibility } from "../contracts/multimodal-routing.js";
import type {
  MediaUnderstandingDecision,
  MediaUnderstandingOutput,
} from "../media-understanding/types.js";
import type { InternalMessageChannel } from "../utils/message-channel.js";
import type { CommandArgs } from "./commands/types.js";

/** Valid message channels for routing. */
export type OriginatingChannelType = ChannelId | InternalMessageChannel;

// ---------------------------------------------------------------------------
// Phase 1: Raw inbound message envelope
// ---------------------------------------------------------------------------

/** Raw message as produced by a channel adapter. */
export type InboundEnvelope = {
  // ---- Body variants ----
  Body?: string;
  /**
   * Agent prompt body (may include envelope/history/context). Prefer this for prompt shaping.
   * Should use real newlines (`\n`), not escaped `\\n`.
   */
  BodyForAgent?: string;
  /**
   * Raw message body without structural context (history, sender labels).
   * Legacy alias for CommandBody. Falls back to Body if not set.
   */
  RawBody?: string;
  /** Prefer for command detection; RawBody is treated as legacy alias. */
  CommandBody?: string;
  /**
   * Command parsing body. Prefer this over CommandBody/RawBody when set.
   * Should be the "clean" text (no history/sender context).
   */
  BodyForCommands?: string;
  CommandArgs?: CommandArgs;

  // ---- Routing / identity ----
  From?: string;
  To?: string;
  /** Provider label (e.g. whatsapp, telegram). */
  Provider?: string;
  /** Provider surface label (e.g. discord, slack). Prefer this over `Provider` when available. */
  Surface?: string;
  ChatType?: string;
  /** Provider account id (multi-account). */
  AccountId?: string;

  // ---- Message IDs ----
  MessageSid?: string;
  /** Provider-specific full message id when MessageSid is a shortened alias. */
  MessageSidFull?: string;
  MessageSids?: string[];
  MessageSidFirst?: string;
  MessageSidLast?: string;

  // ---- Reply context ----
  ReplyToId?: string;
  /** Provider-specific full reply-to id when ReplyToId is a shortened alias. */
  ReplyToIdFull?: string;
  ReplyToBody?: string;
  ReplyToSender?: string;
  ReplyToIsQuote?: boolean;

  // ---- Forwarded context ----
  ForwardedFrom?: string;
  ForwardedFromType?: string;
  ForwardedFromId?: string;
  ForwardedFromUsername?: string;
  ForwardedFromTitle?: string;
  ForwardedFromSignature?: string;
  ForwardedFromChatType?: string;
  ForwardedFromMessageId?: number;
  ForwardedDate?: number;

  // ---- Thread context ----
  ThreadStarterBody?: string;
  /** Full thread history when starting a new thread session. */
  ThreadHistoryBody?: string;
  IsFirstThreadTurn?: boolean;
  ThreadLabel?: string;
  /** Thread identifier (Telegram topic id or Matrix thread event id). */
  MessageThreadId?: string | number;
  /** Telegram forum supergroup marker. */
  IsForum?: boolean;

  // ---- Media ----
  MediaPath?: string;
  MediaUrl?: string;
  MediaType?: string;
  MediaDir?: string;
  MediaPaths?: string[];
  MediaUrls?: string[];
  MediaTypes?: string[];
  /** Telegram sticker metadata (emoji, set name, file IDs, cached description). */
  Sticker?: StickerMetadata;
  OutputDir?: string;
  OutputBase?: string;
  /** Remote host for SCP when media lives on a different machine (e.g., marv@192.168.64.3). */
  MediaRemoteHost?: string;
  Transcript?: string;

  // ---- Sender ----
  SenderName?: string;
  SenderId?: string;
  SenderUsername?: string;
  SenderTag?: string;
  SenderE164?: string;
  Timestamp?: number;

  // ---- Group ----
  /** Human label for envelope headers (conversation label, not sender). */
  ConversationLabel?: string;
  GroupSubject?: string;
  /** Human label for channel-like group conversations (e.g. #general, #support). */
  GroupChannel?: string;
  GroupSpace?: string;
  GroupMembers?: string;
  GroupSystemPrompt?: string;
  WasMentioned?: boolean;

  // ---- Chat history ----
  /**
   * Recent chat history for context (untrusted user content). Prefer passing this
   * as structured context blocks in the user prompt rather than rendering plaintext envelopes.
   */
  InboundHistory?: Array<{
    sender: string;
    body: string;
    timestamp?: number;
  }>;

  // ---- Reply routing ----
  /**
   * Originating channel for reply routing.
   * When set, replies should be routed back to this provider
   * instead of using lastChannel from the session.
   */
  OriginatingChannel?: OriginatingChannelType;
  /**
   * Originating destination for reply routing.
   * The chat/channel/user ID where the reply should be sent.
   */
  OriginatingTo?: string;
};

// ---------------------------------------------------------------------------
// Phase 2: Enriched turn context (after context assembly)
// ---------------------------------------------------------------------------

/** Inbound envelope enriched with routing, session info, understanding results, and command state. */
export type TurnContext = InboundEnvelope & {
  // ---- Session routing ----
  SessionKey?: string;
  ParentSessionKey?: string;

  // ---- Media understanding ----
  PromptMedia?: PromptMediaRef[];
  MultimodalRouting?: MediaPromptCompatibility;
  MediaUnderstanding?: MediaUnderstandingOutput[];
  MediaUnderstandingDecisions?: MediaUnderstandingDecision[];
  LinkUnderstanding?: string[];

  // ---- Command authorization ----
  CommandAuthorized?: boolean;
  CommandSource?: "text" | "native";
  CommandTargetSessionKey?: string;
  /** Gateway client scopes when the message originates from the gateway. */
  GatewayClientScopes?: string[];

  // ---- Prompt shaping ----
  Prompt?: string;
  MaxChars?: number;

  // ---- Untrusted context ----
  /** Untrusted metadata that must not be treated as system instructions. */
  UntrustedContext?: string[];
  /** Explicit owner allowlist overrides (trusted, configuration-derived). */
  OwnerAllowFrom?: Array<string | number>;

  // ---- Hook integration ----
  /**
   * Messages from hooks to be included in the response.
   * Used for hook confirmation messages like "Session context saved to memory".
   */
  HookMessages?: string[];
};

// ---------------------------------------------------------------------------
// Phase 3: Finalized context (CommandAuthorized narrowed to boolean)
// ---------------------------------------------------------------------------

export type FinalizedTurnContext = Omit<TurnContext, "CommandAuthorized"> & {
  /**
   * Always set by finalizeInboundContext().
   * Default-deny: missing/undefined becomes false.
   */
  CommandAuthorized: boolean;
};

// ---------------------------------------------------------------------------
// Phase 4: Template context (session-derived fields for template interpolation)
// ---------------------------------------------------------------------------

export type SessionTemplateContext = TurnContext & {
  BodyStripped?: string;
  SessionId?: string;
  IsNewSession?: string;
};

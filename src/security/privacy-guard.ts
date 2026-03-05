/**
 * Privacy Guard — Context detection for sensitive information protection.
 *
 * Determines whether the current conversation context requires privacy
 * safeguards (e.g. group chats, non-owner senders). When active, the agent
 * MUST NOT reveal secrets, API keys, passwords, or other sensitive data.
 */

export type PrivacyContext = {
  /** Whether the message sender is the bot owner. */
  senderIsOwner: boolean;
  /** Whether this is a multi-user DM (group DM). */
  isMultiUserDm: boolean;
  /** Whether the channel is a group or public channel. */
  isGroupChannel: boolean;
  /** Channel type classification. */
  channelType: "dm" | "group" | "public" | "thread" | "unknown";
  /** Number of recipients who can see the message. */
  recipientCount: number;
};

/**
 * Determine whether the privacy guard should be active for a given context.
 *
 * The guard activates if ANY of the following are true:
 * 1. The sender is not the owner.
 * 2. The conversation is a multi-user DM.
 * 3. The channel is a group or public channel.
 * 4. There are multiple recipients.
 */
export function requiresPrivacyGuard(ctx: PrivacyContext): boolean {
  if (!ctx.senderIsOwner) {
    return true;
  }
  if (ctx.isMultiUserDm) {
    return true;
  }
  if (ctx.isGroupChannel) {
    return true;
  }
  if (ctx.channelType === "group" || ctx.channelType === "public") {
    return true;
  }
  if (ctx.recipientCount > 1) {
    return true;
  }
  return false;
}

/**
 * Build a minimal privacy context from commonly available message metadata.
 */
export function buildPrivacyContext(params: {
  senderIsOwner: boolean;
  isMultiUserDm?: boolean;
  channelType?: PrivacyContext["channelType"];
  recipientCount?: number;
}): PrivacyContext {
  const channelType = params.channelType ?? "unknown";
  const isGroupChannel = channelType === "group" || channelType === "public";
  return {
    senderIsOwner: params.senderIsOwner,
    isMultiUserDm: params.isMultiUserDm ?? false,
    isGroupChannel,
    channelType,
    recipientCount: params.recipientCount ?? (isGroupChannel ? 2 : 1),
  };
}

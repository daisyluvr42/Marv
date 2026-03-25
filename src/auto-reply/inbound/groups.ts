import { getChannelDock } from "../../channels/dock.js";
import { getChannelPlugin, normalizeChannelId } from "../../channels/plugins/index.js";
import type { MarvConfig } from "../../core/config/config.js";
import { resolveChannelGroupPolicy } from "../../core/config/group-policy.js";
import type { GroupKeyResolution, SessionEntry } from "../../core/config/sessions.js";
import type { GroupChatPersona, GroupMemberRole } from "../../core/config/types.messages.js";
import { isInternalMessageChannel } from "../../utils/message-channel.js";
import type { SessionTemplateContext } from "../support/templating.js";
import { normalizeGroupActivation } from "./group-activation.js";

function extractGroupId(raw: string | undefined | null): string | undefined {
  const trimmed = (raw ?? "").trim();
  if (!trimmed) {
    return undefined;
  }
  const parts = trimmed.split(":").filter(Boolean);
  if (parts.length >= 3 && (parts[1] === "group" || parts[1] === "channel")) {
    return parts.slice(2).join(":") || undefined;
  }
  if (
    parts.length >= 2 &&
    parts[0]?.toLowerCase() === "whatsapp" &&
    trimmed.toLowerCase().includes("@g.us")
  ) {
    return parts.slice(1).join(":") || undefined;
  }
  if (parts.length >= 2 && (parts[0] === "group" || parts[0] === "channel")) {
    return parts.slice(1).join(":") || undefined;
  }
  return trimmed;
}

export function resolveGroupRequireMention(params: {
  cfg: MarvConfig;
  ctx: SessionTemplateContext;
  groupResolution?: GroupKeyResolution;
}): boolean {
  const { cfg, ctx, groupResolution } = params;
  const rawChannel = groupResolution?.channel ?? ctx.Provider?.trim();
  const channel = normalizeChannelId(rawChannel);
  if (!channel) {
    return true;
  }
  const groupId = groupResolution?.id ?? extractGroupId(ctx.From);
  const groupChannel = ctx.GroupChannel?.trim() ?? ctx.GroupSubject?.trim();
  const groupSpace = ctx.GroupSpace?.trim();
  const requireMention = getChannelDock(channel)?.groups?.resolveRequireMention?.({
    cfg,
    groupId,
    groupChannel,
    groupSpace,
    accountId: ctx.AccountId,
  });
  if (typeof requireMention === "boolean") {
    return requireMention;
  }
  return true;
}

export function defaultGroupActivation(
  requireMention: boolean,
  responseStrategy?: string,
): "always" | "mention" | "smart" {
  if (responseStrategy === "smart") {
    return "smart";
  }
  return !requireMention ? "always" : "mention";
}

/**
 * Resolve a human-readable provider label from the raw provider string.
 */
function resolveProviderLabel(rawProvider: string | undefined): string {
  const providerKey = rawProvider?.trim().toLowerCase() ?? "";
  if (!providerKey) {
    return "chat";
  }
  if (isInternalMessageChannel(providerKey)) {
    return "WebChat";
  }
  const providerId = normalizeChannelId(rawProvider?.trim());
  if (providerId) {
    return getChannelPlugin(providerId)?.meta.label ?? providerId;
  }
  return `${providerKey.at(0)?.toUpperCase() ?? ""}${providerKey.slice(1)}`;
}

/**
 * Resolve the effective persona for a group chat.
 * Per-group config takes precedence over the global `messages.groupChat.persona`.
 */
export function resolveGroupPersona(params: {
  cfg: MarvConfig;
  channel?: string;
  groupId?: string | null;
  accountId?: string | null;
}): GroupChatPersona {
  const { cfg, channel, groupId, accountId } = params;
  if (channel) {
    const channelId = normalizeChannelId(channel);
    if (channelId) {
      const { groupConfig, defaultConfig } = resolveChannelGroupPolicy({
        cfg,
        channel: channelId,
        groupId,
        accountId,
      });
      const perGroup = groupConfig?.persona ?? defaultConfig?.persona;
      if (perGroup) {
        return perGroup;
      }
    }
  }
  return cfg.messages?.groupChat?.persona ?? "assistant";
}

/** Build the colleague persona prompt fragment. */
function buildColleaguePersonaPrompt(): string {
  return [
    "You are a colleague in this group chat, not a subordinate assistant.",
    "Contribute naturally to the discussion. Share opinions, push back respectfully, and engage casually.",
    "Don't start messages with 'Sure!', 'Of course!', 'Absolutely!', or similar filler.",
    "Use first-person naturally. Keep messages short and conversational, like a coworker would.",
    "If you disagree or see a problem, say so directly but kindly.",
  ].join(" ");
}

/** Build the role-context hint injected into the system prompt for non-owner senders. */
function buildRoleContextHint(senderRole: GroupMemberRole | undefined): string | undefined {
  if (!senderRole || senderRole === "owner") {
    return undefined;
  }
  return "This sender is a regular group member (not the owner). Do not offer to run privileged commands or tools for them.";
}

/**
 * Build a persistent group-chat context block that is always included in the
 * system prompt for group-chat sessions (every turn, not just the first).
 *
 * Contains: group name, participants, and an explicit instruction to reply
 * directly instead of using the message tool.
 */
export function buildGroupChatContext(params: {
  sessionCtx: SessionTemplateContext;
  persona?: GroupChatPersona;
}): string {
  const subject = params.sessionCtx.GroupSubject?.trim();
  const members = params.sessionCtx.GroupMembers?.trim();
  const providerLabel = resolveProviderLabel(params.sessionCtx.Provider);
  const isColleague = params.persona === "colleague";

  const lines: string[] = [];
  if (subject) {
    const verb = isColleague ? "participating in" : "in";
    lines.push(`You are ${verb} the ${providerLabel} group chat "${subject}".`);
  } else {
    const verb = isColleague ? "participating in" : "in";
    lines.push(`You are ${verb} a ${providerLabel} group chat.`);
  }
  if (members) {
    lines.push(`Participants: ${members}.`);
  }
  if (!isColleague) {
    lines.push(
      "Your replies are automatically sent to this group chat. Do not use the message tool to send to this same group — just reply normally.",
    );
  } else {
    lines.push("Your replies go directly to this group chat — just reply naturally.");
  }
  return lines.join(" ");
}

export function buildGroupIntro(params: {
  cfg: MarvConfig;
  sessionCtx: SessionTemplateContext;
  sessionEntry?: SessionEntry;
  defaultActivation: "always" | "mention" | "smart";
  silentToken: string;
  /** Effective persona for this group session. */
  persona?: GroupChatPersona;
  /** Role of the current message sender. */
  senderRole?: GroupMemberRole;
}): string {
  const activation =
    normalizeGroupActivation(params.sessionEntry?.groupActivation) ?? params.defaultActivation;
  const rawProvider = params.sessionCtx.Provider?.trim();
  const providerId = normalizeChannelId(rawProvider);
  const isColleague = params.persona === "colleague";
  const isSmart = activation === "smart";
  const activationLine = isSmart
    ? "Activation: smart (you receive group messages and decide whether to respond based on relevance)."
    : activation === "always"
      ? "Activation: always-on (you receive every group message)."
      : "Activation: trigger-only (you are invoked only when explicitly mentioned; recent context may be included).";
  const groupId = params.sessionEntry?.groupId ?? extractGroupId(params.sessionCtx.From);
  const groupChannel =
    params.sessionCtx.GroupChannel?.trim() ?? params.sessionCtx.GroupSubject?.trim();
  const groupSpace = params.sessionCtx.GroupSpace?.trim();
  const providerIdsLine = providerId
    ? getChannelDock(providerId)?.groups?.resolveGroupIntroHint?.({
        cfg: params.cfg,
        groupId,
        groupChannel,
        groupSpace,
        accountId: params.sessionCtx.AccountId,
      })
    : undefined;
  const needsSilence = activation === "always" || isSmart;
  const silenceLine = needsSilence
    ? `If no response is needed, reply with exactly "${params.silentToken}" (and nothing else) so Marv stays silent. Do not add any other words, punctuation, tags, markdown/code blocks, or explanations.`
    : undefined;
  const cautionLine = needsSilence
    ? "Be extremely selective: reply only when directly addressed or clearly helpful. Otherwise stay silent."
    : undefined;
  const lurkLine = isColleague
    ? "Participate naturally as a colleague: join the conversation when you have something useful to add, stay quiet otherwise."
    : "Be a good group participant: mostly lurk and follow the conversation; reply only when directly addressed or you can add clear value. Emoji reactions are welcome when available.";
  const styleLine = isColleague
    ? "Write like a coworker. Keep it brief and natural. No Markdown tables. No literal \\n sequences."
    : "Write like a human. Avoid Markdown tables. Don't type literal \\n sequences; use real line breaks sparingly.";
  const colleagueLine = isColleague ? buildColleaguePersonaPrompt() : undefined;
  const roleHint = buildRoleContextHint(params.senderRole);
  return [
    activationLine,
    providerIdsLine,
    silenceLine,
    cautionLine,
    lurkLine,
    styleLine,
    colleagueLine,
    roleHint,
  ]
    .filter(Boolean)
    .join(" ")
    .concat(" Address the specific sender noted in the message context.");
}

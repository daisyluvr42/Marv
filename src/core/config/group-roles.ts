import type { ChannelId } from "../../channels/plugins/types.js";
import { normalizeAnyChannelId } from "../../channels/registry.js";
import type { MarvConfig } from "./config.js";
import type { GroupToolPolicySender } from "./group-policy.js";
import type { GroupMemberRole } from "./types.messages.js";

/**
 * Resolve the group-chat role for a given sender.
 *
 * Returns "owner" when the sender matches `commands.ownerAllowFrom` or the
 * channel's `allowFrom` (and no owner list is explicitly configured).
 * Otherwise returns "member".
 */
export function resolveGroupMemberRole(params: {
  cfg: MarvConfig;
  providerId?: ChannelId;
  channelAllowFrom?: Array<string | number>;
  sender: GroupToolPolicySender;
}): GroupMemberRole {
  const { cfg, providerId, channelAllowFrom, sender } = params;
  const ownerAllowFrom = cfg.commands?.ownerAllowFrom;

  const senderCandidates = buildSenderCandidates(sender);
  if (senderCandidates.length === 0) {
    return "member";
  }

  // Check explicit ownerAllowFrom first.
  if (Array.isArray(ownerAllowFrom) && ownerAllowFrom.length > 0) {
    for (const entry of ownerAllowFrom) {
      const trimmed = String(entry ?? "").trim();
      if (!trimmed) {
        continue;
      }

      // Handle channel-prefixed entries like "telegram:123456".
      const separatorIndex = trimmed.indexOf(":");
      if (separatorIndex > 0) {
        const prefix = trimmed.slice(0, separatorIndex);
        const channel = normalizeAnyChannelId(prefix);
        if (channel) {
          if (providerId && channel !== providerId) {
            continue;
          }
          const remainder = trimmed.slice(separatorIndex + 1).trim();
          if (remainder && matchesSender(remainder, senderCandidates)) {
            return "owner";
          }
          continue;
        }
      }

      if (trimmed === "*" || matchesSender(trimmed, senderCandidates)) {
        return "owner";
      }
    }
    return "member";
  }

  // Fall back to channel allowFrom (non-wildcard entries are treated as owners).
  if (Array.isArray(channelAllowFrom) && channelAllowFrom.length > 0) {
    const nonWildcard = channelAllowFrom.map((e) => String(e).trim()).filter((e) => e && e !== "*");
    if (nonWildcard.length === 0) {
      // Wildcard-only allowFrom — everyone is a member by default.
      return "member";
    }
    for (const entry of nonWildcard) {
      if (matchesSender(entry, senderCandidates)) {
        return "owner";
      }
    }
  }

  return "member";
}

function buildSenderCandidates(sender: GroupToolPolicySender): string[] {
  const candidates: string[] = [];
  const push = (v?: string | null) => {
    const t = v?.trim();
    if (t) {
      candidates.push(t.toLowerCase());
    }
  };
  push(sender.senderId);
  push(sender.senderE164);
  push(sender.senderUsername);
  push(sender.senderName);
  return candidates;
}

function matchesSender(entry: string, candidates: string[]): boolean {
  const normalized = entry.toLowerCase();
  return candidates.includes(normalized);
}

import type { MarvConfig } from "../../core/config/config.js";
import { createSubsystemLogger } from "../../logging/subsystem.js";
import { parseAgentSessionKey, resolveAgentIdFromSessionKey } from "../../routing/session-key.js";
import { isSmallTalk } from "../small-talk.js";
import { ingestSoulMemoryEvent } from "./soul-memory-store.js";

const log = createSubsystemLogger("memory/runtime-ingest");

function isEnabled(cfg?: MarvConfig): boolean {
  if (!cfg) {
    return true;
  }
  return cfg.memory?.runtimeIngest !== false;
}

function isRelationshipTurn(content: string): boolean {
  return isSmallTalk(content);
}

function buildScope(sessionKey: string | undefined, agentId: string) {
  const parsed = parseAgentSessionKey(sessionKey);
  if (parsed?.rest) {
    return {
      scopeType: "session",
      scopeId: parsed.rest,
    };
  }
  return {
    scopeType: "agent",
    scopeId: agentId,
  };
}

export function ingestInboundMessageToSoulMemory(params: {
  cfg?: MarvConfig;
  sessionKey?: string;
  content: string;
  channelId?: string;
  conversationId?: string;
  accountId?: string;
  messageId?: string;
  nowMs?: number;
}): void {
  const content = params.content.trim();
  if (!isEnabled(params.cfg) || !content) {
    return;
  }
  const agentId = resolveAgentIdFromSessionKey(params.sessionKey) || "main";
  const scope = buildScope(params.sessionKey, agentId);
  const relationship = isRelationshipTurn(content);
  try {
    ingestSoulMemoryEvent({
      agentId,
      scopeType: scope.scopeType,
      scopeId: scope.scopeId,
      kind: relationship ? "relationship_chat" : "user_turn",
      content,
      summary: relationship ? `Relationship memory: ${content}` : `User said: ${content}`,
      source: "runtime_event",
      recordKind: relationship ? "relationship" : "fact",
      nowMs: params.nowMs,
      skipArchive: true,
      metadata: {
        sessionKey: params.sessionKey,
        channelId: params.channelId,
        conversationId: params.conversationId,
        accountId: params.accountId,
        messageId: params.messageId,
        role: "user",
      },
    });
  } catch (err) {
    log.debug(`Failed inbound structured memory ingestion: ${String(err)}`);
  }
}

export function ingestOutboundMessageToSoulMemory(params: {
  cfg?: MarvConfig;
  sessionKey?: string;
  content: string;
  channelId?: string;
  conversationId?: string;
  accountId?: string;
  messageId?: string;
  nowMs?: number;
}): void {
  const content = params.content.trim();
  if (!isEnabled(params.cfg) || !content) {
    return;
  }
  const agentId = resolveAgentIdFromSessionKey(params.sessionKey) || "main";
  const scope = buildScope(params.sessionKey, agentId);
  const relationship = isRelationshipTurn(content);
  try {
    ingestSoulMemoryEvent({
      agentId,
      scopeType: scope.scopeType,
      scopeId: scope.scopeId,
      kind: relationship ? "relationship_reply" : "assistant_turn",
      content,
      summary: relationship ? `Relationship reply: ${content}` : `Assistant replied: ${content}`,
      source: "runtime_event",
      recordKind: relationship ? "relationship" : "fact",
      nowMs: params.nowMs,
      skipArchive: true,
      metadata: {
        sessionKey: params.sessionKey,
        channelId: params.channelId,
        conversationId: params.conversationId,
        accountId: params.accountId,
        messageId: params.messageId,
        role: "assistant",
      },
    });
  } catch (err) {
    log.debug(`Failed outbound structured memory ingestion: ${String(err)}`);
  }
}

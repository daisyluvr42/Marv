import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { callGateway } from "../core/gateway/call.js";
import { extractAssistantText, stripToolMessages } from "./tools/sessions/sessions-helpers.js";

/**
 * Specification for context to inject into a subagent task prompt.
 * Allows the parent to selectively share relevant conversation history,
 * tool results, and file contents with the subagent.
 */
export type SubagentContextSpec = {
  /** Include the most recent N conversation turns from the parent session. */
  recentTurns?: number;
  /** Include tool result content from these tool names only. */
  includeToolResults?: string[];
  /** Include file snippets from these workspace-relative paths. */
  includeFiles?: string[];
  /** Maximum total characters for the assembled context block (default: 8000). */
  maxContextChars?: number;
  /** Custom preamble prepended to the context block. */
  preamble?: string;
};

const DEFAULT_MAX_CONTEXT_CHARS = 8_000;
const MAX_FILE_SNIPPET_CHARS = 2_000;

/**
 * Build a context block string from the given spec.
 *
 * Returns an empty string if the spec produces no content.
 */
export async function buildSubagentContext(params: {
  spec: SubagentContextSpec;
  parentSessionKey: string;
  workspaceDir?: string;
}): Promise<string> {
  const { spec, parentSessionKey, workspaceDir } = params;
  const maxChars = spec.maxContextChars ?? DEFAULT_MAX_CONTEXT_CHARS;
  const blocks: string[] = [];

  if (spec.preamble?.trim()) {
    blocks.push(spec.preamble.trim());
  }

  // --- Recent conversation turns ---
  const wantTurns = spec.recentTurns ?? 0;
  const wantToolResults = spec.includeToolResults;
  if (wantTurns > 0 || (wantToolResults && wantToolResults.length > 0)) {
    const historyBlock = await buildHistoryBlock({
      parentSessionKey,
      recentTurns: wantTurns,
      includeToolResults: wantToolResults,
    });
    if (historyBlock) {
      blocks.push(historyBlock);
    }
  }

  // --- File snippets ---
  if (spec.includeFiles && spec.includeFiles.length > 0 && workspaceDir) {
    const fileBlock = await buildFileBlock({
      files: spec.includeFiles,
      workspaceDir,
    });
    if (fileBlock) {
      blocks.push(fileBlock);
    }
  }

  if (blocks.length === 0) {
    return "";
  }

  const assembled = blocks.join("\n\n---\n\n");
  return truncateToLimit(assembled, maxChars);
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

async function buildHistoryBlock(params: {
  parentSessionKey: string;
  recentTurns: number;
  includeToolResults?: string[];
}): Promise<string | undefined> {
  // Fetch enough messages to cover N turns (each turn = user + assistant ≈ 2 messages).
  const limit = Math.max(params.recentTurns * 2, 20);
  let history: { messages: unknown[] } | undefined;
  try {
    history = await callGateway<{ messages: unknown[] }>({
      method: "chat.history",
      params: {
        sessionKey: params.parentSessionKey,
        limit,
      },
      timeoutMs: 10_000,
    });
  } catch {
    return undefined;
  }
  const allMessages = Array.isArray(history?.messages) ? history.messages : [];
  if (allMessages.length === 0) {
    return undefined;
  }

  const parts: string[] = [];

  // Extract recent turns.
  if (params.recentTurns > 0) {
    const cleaned = stripToolMessages(allMessages);
    const recent = cleaned.slice(-params.recentTurns * 2);
    const turnLines: string[] = [];
    for (const msg of recent) {
      const role = (msg as { role?: string }).role;
      if (role === "assistant") {
        const text = extractAssistantText(msg);
        if (text) {
          turnLines.push(`**assistant**: ${text}`);
        }
      } else if (role === "user") {
        const content = (msg as { content?: unknown }).content;
        const text = extractUserText(content);
        if (text) {
          turnLines.push(`**user**: ${text}`);
        }
      }
    }
    if (turnLines.length > 0) {
      parts.push(`## Recent conversation\n\n${turnLines.join("\n\n")}`);
    }
  }

  // Extract tool results by name.
  if (params.includeToolResults && params.includeToolResults.length > 0) {
    const toolNames = new Set(params.includeToolResults.map((n) => n.toLowerCase()));
    const toolLines: string[] = [];
    for (const msg of allMessages) {
      const role = (msg as { role?: string }).role;
      if (role === "toolResult" || role === "tool") {
        const name = ((msg as { name?: string }).name ?? "").toLowerCase();
        if (toolNames.has(name)) {
          const content = (msg as { content?: unknown }).content;
          const text = extractToolResultText(content);
          if (text) {
            toolLines.push(`### ${name}\n\`\`\`\n${text.slice(0, 1500)}\n\`\`\``);
          }
        }
      }
    }
    if (toolLines.length > 0) {
      parts.push(`## Relevant tool results\n\n${toolLines.join("\n\n")}`);
    }
  }

  return parts.length > 0 ? parts.join("\n\n") : undefined;
}

async function buildFileBlock(params: {
  files: string[];
  workspaceDir: string;
}): Promise<string | undefined> {
  const snippets: string[] = [];
  for (const filePath of params.files.slice(0, 10)) {
    try {
      const absPath = resolve(params.workspaceDir, filePath);
      const content = await readFile(absPath, "utf-8");
      const truncated = content.slice(0, MAX_FILE_SNIPPET_CHARS);
      const suffix = content.length > MAX_FILE_SNIPPET_CHARS ? "\n... (truncated)" : "";
      snippets.push(`### ${filePath}\n\`\`\`\n${truncated}${suffix}\n\`\`\``);
    } catch {
      // File not found or unreadable — skip silently.
    }
  }
  if (snippets.length === 0) {
    return undefined;
  }
  return `## File context\n\n${snippets.join("\n\n")}`;
}

function extractUserText(content: unknown): string | undefined {
  if (typeof content === "string") {
    return content.trim() || undefined;
  }
  if (Array.isArray(content)) {
    const texts: string[] = [];
    for (const part of content) {
      if (part && typeof part === "object" && (part as { type?: string }).type === "text") {
        const t = (part as { text?: string }).text;
        if (t) {
          texts.push(t);
        }
      }
    }
    return texts.join(" ").trim() || undefined;
  }
  return undefined;
}

function extractToolResultText(content: unknown): string | undefined {
  if (typeof content === "string") {
    return content.trim() || undefined;
  }
  if (Array.isArray(content)) {
    const texts: string[] = [];
    for (const part of content) {
      if (part && typeof part === "object") {
        const t = (part as { text?: string }).text;
        if (t) {
          texts.push(t);
        }
      }
    }
    return texts.join("\n").trim() || undefined;
  }
  return undefined;
}

function truncateToLimit(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, maxChars)}\n\n... (context truncated to ${maxChars} chars)`;
}

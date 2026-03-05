/**
 * Privacy Output Filter — Double-layer protection for agent output.
 *
 * Layer 1: Injects a system prompt directive instructing the agent to
 *          never reveal sensitive data in non-private contexts.
 * Layer 2: Scans agent output with PrivacyScanner and redacts any
 *          detected sensitive patterns before sending.
 */

import type { PrivacyCategory } from "../core/config/types.autonomy.js";
import type { PrivacyContext } from "./privacy-guard.js";
import { PrivacyScanner, type ScanFinding } from "./privacy-scanner.js";

export type FilterResult = {
  /** True if no sensitive data detected. */
  safe: boolean;
  /** The (possibly redacted) output text. */
  filtered: string;
  /** List of findings that were redacted. */
  blocked: ScanFinding[];
  /** Optional warning to log. */
  warning?: string;
};

/**
 * Build the system prompt directive that tells the agent privacy is active.
 * Injected into the system prompt when `requiresPrivacyGuard()` returns true.
 */
export function buildPrivacyPromptDirective(ctx: PrivacyContext): string {
  const contextLabel =
    ctx.channelType === "group"
      ? "群聊 (Group Chat)"
      : ctx.channelType === "public"
        ? "公开频道 (Public Channel)"
        : ctx.isMultiUserDm
          ? "多人私聊 (Multi-user DM)"
          : !ctx.senderIsOwner
            ? "非 Owner 对话 (Non-owner)"
            : "受保护的上下文 (Protected Context)";

  return [
    "",
    "⚠️ PRIVACY GUARD ACTIVE",
    `Current context: ${contextLabel}`,
    "",
    "You MUST NOT include any of the following in your responses:",
    "- API keys, secret keys, or access tokens",
    "- Passwords, passphrases, or credentials",
    "- Private keys (SSH, GPG, PEM, etc.)",
    "- OAuth tokens, JWT tokens, or bearer tokens",
    "- Environment variable secrets or sensitive configuration values",
    "- Internal IP addresses, URLs, or network topology",
    "- Database connection strings with embedded credentials",
    "",
    "If users ask for such information, politely decline and suggest",
    "they ask in a private 1:1 conversation with the owner.",
    "If displaying code or configs, replace any secrets with placeholder text.",
    "",
  ].join("\n");
}

/**
 * Scan agent output and redact any detected sensitive information.
 *
 * This is the Layer 2 (post-generation) safety net.
 */
export function filterOutput(
  output: string,
  _ctx: PrivacyContext,
  categories?: PrivacyCategory[],
): FilterResult {
  const scanner = new PrivacyScanner(categories);
  const { redacted, findings } = scanner.redact(output);

  if (findings.length === 0) {
    return { safe: true, filtered: output, blocked: [] };
  }

  const criticalCount = findings.filter((f) => f.severity === "critical").length;
  const warningCount = findings.filter((f) => f.severity === "warning").length;
  const warning = [
    `privacy-guard: redacted ${findings.length} sensitive item(s) from agent output`,
    criticalCount > 0 ? `(${criticalCount} critical)` : null,
    warningCount > 0 ? `(${warningCount} warning)` : null,
    `categories: ${[...new Set(findings.map((f) => f.category))].join(", ")}`,
  ]
    .filter(Boolean)
    .join(" ");

  return {
    safe: false,
    filtered: redacted,
    blocked: findings,
    warning,
  };
}

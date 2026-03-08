import type { Bot } from "grammy";
import { InlineKeyboard } from "grammy";
import type { MarvConfig } from "../../../core/config/config.js";
import { buildGatewayConnectionDetails } from "../../../core/gateway/call.js";
import { GatewayClient } from "../../../core/gateway/client.js";
import type { EventFrame } from "../../../core/gateway/protocol/index.js";
import type {
  ExecApprovalDecision,
  ExecApprovalRequest,
  ExecApprovalResolved,
} from "../../../infra/exec-approvals.js";
import { logDebug, logError } from "../../../logger.js";
import { normalizeAccountId } from "../../../routing/session-key.js";
import type { RuntimeEnv } from "../../../runtime.js";
import { GATEWAY_CLIENT_MODES, GATEWAY_CLIENT_NAMES } from "../../../utils/message-channel.js";

const EXEC_APPROVAL_KEY = "execapprv"; // Shortened to fit Telegram's 64-byte callback_data limit

export type { ExecApprovalRequest, ExecApprovalResolved };

/** Escape HTML special chars for Telegram HTML parse mode. */
function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function extractTelegramChatId(sessionKey?: string | null): string | null {
  if (!sessionKey) {
    return null;
  }
  const match = sessionKey.match(/telegram:(?:direct|group):(-?\d+)/);
  return match ? match[1] : null;
}

type PendingApproval = {
  chatId: number;
  messageId: number;
  timeoutId: NodeJS.Timeout;
};

// Telegram callback_data limit is 64 bytes. We must keep this compact.
export function buildExecApprovalCallbackData(
  approvalId: string,
  action: ExecApprovalDecision,
): string {
  // Format: "execapprv:<shortId>:<act>"
  // action can be: "once" (allow-once), "always" (allow-always), "deny" (deny)
  const act = action === "allow-once" ? "1" : action === "allow-always" ? "A" : "0";
  // Assuming approvalId is a UUID length 36.
  // "execapprv:uuid:1" -> 9 + 1 + 36 + 1 + 1 = 48 bytes. Fits easily.
  return `${EXEC_APPROVAL_KEY}:${approvalId}:${act}`;
}

export function parseExecApprovalData(
  data: string,
): { approvalId: string; action: ExecApprovalDecision } | null {
  if (!data.startsWith(`${EXEC_APPROVAL_KEY}:`)) {
    return null;
  }
  const parts = data.split(":");
  if (parts.length !== 3) {
    return null;
  }
  const actChar = parts[2];
  let action: ExecApprovalDecision;
  if (actChar === "1") {
    action = "allow-once";
  } else if (actChar === "A") {
    action = "allow-always";
  } else if (actChar === "0") {
    action = "deny";
  } else {
    return null;
  }

  return { approvalId: parts[1], action };
}

export type TelegramExecApprovalHandlerOpts = {
  bot: Bot;
  accountId: string;
  config: import("../../../core/config/types.telegram.js").TelegramExecApprovalConfig | undefined;
  gatewayUrl?: string;
  cfg: MarvConfig;
  runtime?: RuntimeEnv;
};

export class TelegramExecApprovalHandler {
  private gatewayClient: GatewayClient | null = null;
  private pending = new Map<string, PendingApproval>();
  private requestCache = new Map<string, ExecApprovalRequest>();
  private opts: TelegramExecApprovalHandlerOpts;
  private started = false;

  constructor(opts: TelegramExecApprovalHandlerOpts) {
    this.opts = opts;
  }

  shouldHandle(_request: ExecApprovalRequest): boolean {
    const config = this.opts.config;
    if (!config?.enabled) {
      return false;
    }
    // We assume if telegram approvals are enabled in config, we process it locally.
    // However, we should check if the sessionKey belongs to this account.
    const requestAccountId = "default"; // Simplified for now
    if (normalizeAccountId(requestAccountId) !== normalizeAccountId(this.opts.accountId)) {
      // return false;
    }
    return true; // Expand logic as needed later, matching discord.
  }

  async start(): Promise<void> {
    if (this.started) {
      return;
    }
    this.started = true;

    const config = this.opts.config;
    if (!config?.enabled) {
      logDebug("telegram exec approvals: disabled");
      return;
    }

    logDebug("telegram exec approvals: starting handler");

    const { url: gatewayUrl } = buildGatewayConnectionDetails({
      config: this.opts.cfg,
      url: this.opts.gatewayUrl,
    });

    this.gatewayClient = new GatewayClient({
      url: gatewayUrl,
      clientName: GATEWAY_CLIENT_NAMES.GATEWAY_CLIENT,
      clientDisplayName: "Telegram Exec Approvals",
      mode: GATEWAY_CLIENT_MODES.BACKEND,
      scopes: ["operator.approvals"],
      onEvent: (evt) => this.handleGatewayEvent(evt),
      onHelloOk: () => {
        logDebug("telegram exec approvals: connected to gateway");
      },
      onConnectError: (err) => {
        logError(`telegram exec approvals: connect error: ${err.message}`);
      },
      onClose: (code, reason) => {
        logDebug(`telegram exec approvals: gateway closed: ${code} ${reason}`);
      },
    });

    this.gatewayClient.start();
  }

  async stop(): Promise<void> {
    if (!this.started) {
      return;
    }
    this.started = false;

    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeoutId);
    }
    this.pending.clear();
    this.requestCache.clear();

    this.gatewayClient?.stop();
    this.gatewayClient = null;

    logDebug("telegram exec approvals: stopped");
  }

  private async handleGatewayEvent(evt: EventFrame): Promise<void> {
    if (evt.event === "exec.approval.requested") {
      await this.handleApprovalRequested(evt.payload as ExecApprovalRequest);
    } else if (evt.event === "exec.approval.resolved") {
      await this.handleApprovalResolved(evt.payload as ExecApprovalResolved);
    }
  }

  private async handleApprovalRequested(request: ExecApprovalRequest): Promise<void> {
    if (!this.shouldHandle(request)) {
      return;
    }

    const approvalId = request.id;
    this.requestCache.set(approvalId, request);

    // Find target chat id.
    const chatIdStr = extractTelegramChatId(request.request.sessionKey);
    let targetChatId: string | number | null = chatIdStr ? parseInt(chatIdStr, 10) : null;
    if (!targetChatId && this.opts.config?.approvers && this.opts.config.approvers.length > 0) {
      // If we couldn't parse it from session key, just send to the first approver ID
      targetChatId = this.opts.config.approvers[0] ?? null;
    }
    if (!targetChatId) {
      logError(
        `telegram exec approvals: could not resolve target chat ID for request ${approvalId}`,
      );
      return;
    }

    const commandText = request.request.command;
    const commandPreview =
      commandText.length > 500 ? `${commandText.slice(0, 500)}...` : commandText;

    // Formatting the message (HTML parse mode avoids MarkdownV2 escaping pitfalls)
    const escapedPreview = escapeHtml(commandPreview);
    const msgText = `⚠️ <b>Exec Approval Required</b>\n\nThe agent is attempting to run a potentially dangerous command.\n\n<pre>${escapedPreview}</pre>\n\nID: <code>${approvalId}</code>`;

    const inlineKeyboard = new InlineKeyboard()
      .text("✅ Allow Once", buildExecApprovalCallbackData(approvalId, "allow-once"))
      .text("🟩 Allow Always", buildExecApprovalCallbackData(approvalId, "allow-always"))
      .row()
      .text("🛑 Deny", buildExecApprovalCallbackData(approvalId, "deny"));

    try {
      const message = await this.opts.bot.api.sendMessage(targetChatId, msgText, {
        parse_mode: "HTML",
        reply_markup: inlineKeyboard,
      });

      const expiresAtMs = request.expiresAtMs;
      const now = Date.now();
      const delayMs = Math.max(0, expiresAtMs - now);

      const timeoutId = setTimeout(() => {
        this.handleApprovalTimeout(approvalId).catch((err) => {
          logError(`telegram exec approvals timeout error: ${String(err)}`);
        });
      }, delayMs);

      this.pending.set(approvalId, {
        chatId: message.chat.id,
        messageId: message.message_id,
        timeoutId,
      });
    } catch (err) {
      logError(`telegram exec approvals: failed to send approval message: ${String(err)}`);
    }
  }

  private async handleApprovalResolved(resolved: ExecApprovalResolved): Promise<void> {
    const approvalId = resolved.id;
    this.requestCache.delete(approvalId);
    const pending = this.pending.get(approvalId);
    if (!pending) {
      return;
    }

    this.pending.delete(approvalId);
    clearTimeout(pending.timeoutId);

    // Update the message in Telegram to reflect it's been resolved
    try {
      let icon = "✅";
      if (resolved.decision === "deny") {
        icon = "🛑";
      }

      const newText = `<b>Exec Approval Resolved</b> ${icon}\n\nDecision: <code>${resolved.decision}</code>\n\nID: <code>${approvalId}</code>`;

      await this.opts.bot.api.editMessageText(pending.chatId, pending.messageId, newText, {
        parse_mode: "HTML",
        reply_markup: { inline_keyboard: [] }, // Remove buttons
      });
    } catch (err) {
      logError(`telegram exec approvals: failed to update resolved message: ${String(err)}`);
    }
  }

  private async handleApprovalTimeout(approvalId: string): Promise<void> {
    this.requestCache.delete(approvalId);
    const pending = this.pending.get(approvalId);
    if (!pending) {
      return;
    }

    this.pending.delete(approvalId);

    try {
      const newText = `⏳ <b>Exec Approval Expired</b>\n\nThis approval request has timed out.\n\nID: <code>${approvalId}</code>`;
      await this.opts.bot.api.editMessageText(pending.chatId, pending.messageId, newText, {
        parse_mode: "HTML",
        reply_markup: { inline_keyboard: [] },
      });
    } catch (err) {
      logError(`telegram exec approvals: failed to update expired message: ${String(err)}`);
    }
  }

  async resolveApproval(approvalId: string, action: ExecApprovalDecision): Promise<void> {
    if (!this.gatewayClient) {
      logError("telegram exec approvals: gateway client not connected");
      return;
    }

    // Resolving manually from inline keyboard click
    const pending = this.pending.get(approvalId);
    if (pending) {
      clearTimeout(pending.timeoutId);
      this.pending.delete(approvalId);
    }

    try {
      await this.gatewayClient.request("exec.approval.resolve", {
        id: approvalId,
        decision: action,
      });
    } catch (err) {
      logError(`telegram exec approvals: failed to resolve via gateway: ${String(err)}`);
      throw err;
    }
  }
}

import type { Bot } from "grammy";
import type { MarvConfig } from "../../../core/config/config.js";
import type { TelegramExecApprovalConfig } from "../../../core/config/types.telegram.js";
import { buildGatewayConnectionDetails } from "../../../core/gateway/call.js";
import { GatewayClient } from "../../../core/gateway/client.js";
import type { EventFrame } from "../../../core/gateway/protocol/index.js";
import { logDebug, logError } from "../../../logger.js";
import type { RuntimeEnv } from "../../../runtime.js";
import { compileSafeRegex } from "../../../security/safe-regex.js";
import { GATEWAY_CLIENT_MODES, GATEWAY_CLIENT_NAMES } from "../../../utils/message-channel.js";

type CronMutationAction = "added" | "updated" | "removed";

export type CronMutationEvent = {
  jobId: string;
  jobName?: string;
  action: CronMutationAction;
  agentId?: string;
  sessionKey?: string;
  sessionTarget?: string;
  deliveryMode?: string;
  nextRunAtMs?: number;
};

export type TelegramCronMutationHandlerOpts = {
  bot: Bot;
  config: TelegramExecApprovalConfig | undefined;
  gatewayUrl?: string;
  cfg: MarvConfig;
  runtime?: RuntimeEnv;
};

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseCronMutationEvent(payload: unknown): CronMutationEvent | null {
  if (!isRecord(payload)) {
    return null;
  }
  const action = payload.action;
  const jobId = typeof payload.jobId === "string" ? payload.jobId.trim() : "";
  if ((action !== "added" && action !== "updated" && action !== "removed") || !jobId) {
    return null;
  }
  const nextRunAtMs =
    typeof payload.nextRunAtMs === "number" && Number.isFinite(payload.nextRunAtMs)
      ? payload.nextRunAtMs
      : undefined;
  return {
    jobId,
    action,
    jobName: typeof payload.jobName === "string" ? payload.jobName.trim() || undefined : undefined,
    agentId: typeof payload.agentId === "string" ? payload.agentId.trim() || undefined : undefined,
    sessionKey:
      typeof payload.sessionKey === "string" ? payload.sessionKey.trim() || undefined : undefined,
    sessionTarget:
      typeof payload.sessionTarget === "string"
        ? payload.sessionTarget.trim() || undefined
        : undefined,
    deliveryMode:
      typeof payload.deliveryMode === "string"
        ? payload.deliveryMode.trim() || undefined
        : undefined,
    nextRunAtMs,
  };
}

function actionLabel(action: CronMutationAction): string {
  if (action === "added") {
    return "added";
  }
  if (action === "updated") {
    return "updated";
  }
  return "removed";
}

function buildCronMutationMessage(event: CronMutationEvent): string {
  const lines = [
    `<b>Cron job ${escapeHtml(actionLabel(event.action))}</b>`,
    escapeHtml(event.jobName || event.jobId),
    "",
    `Job: <code>${escapeHtml(event.jobId)}</code>`,
  ];
  if (event.agentId) {
    lines.push(`Agent: <code>${escapeHtml(event.agentId)}</code>`);
  }
  if (event.sessionTarget) {
    lines.push(`Target: <code>${escapeHtml(event.sessionTarget)}</code>`);
  }
  if (event.deliveryMode) {
    lines.push(`Delivery: <code>${escapeHtml(event.deliveryMode)}</code>`);
  }
  if (event.nextRunAtMs) {
    lines.push(`Next: <code>${escapeHtml(new Date(event.nextRunAtMs).toISOString())}</code>`);
  }
  if (event.sessionKey) {
    lines.push(`Session: <code>${escapeHtml(event.sessionKey)}</code>`);
  }
  return lines.join("\n");
}

export class TelegramCronMutationHandler {
  private gatewayClient: GatewayClient | null = null;
  private started = false;

  constructor(private opts: TelegramCronMutationHandlerOpts) {}

  shouldHandle(event: CronMutationEvent): boolean {
    const config = this.opts.config;
    if (!config?.approvers?.length) {
      return false;
    }
    if (config.agentFilter?.length) {
      const agentId = event.agentId?.trim() || "";
      if (!agentId || !config.agentFilter.includes(agentId)) {
        return false;
      }
    }
    if (config.sessionFilter?.length) {
      const session = event.sessionKey?.trim() || "";
      if (!session) {
        return false;
      }
      const matches = config.sessionFilter.some((pattern) => {
        const compiled = compileSafeRegex(pattern);
        if (!compiled) {
          return session.includes(pattern);
        }
        return session.includes(pattern) || compiled.test(session);
      });
      if (!matches) {
        return false;
      }
    }
    return true;
  }

  async start(): Promise<void> {
    if (this.started) {
      return;
    }
    this.started = true;

    if (!this.opts.config?.approvers?.length) {
      logDebug("telegram cron notifications: no approvers configured");
      return;
    }

    const { url: gatewayUrl } = buildGatewayConnectionDetails({
      config: this.opts.cfg,
      url: this.opts.gatewayUrl,
    });

    this.gatewayClient = new GatewayClient({
      url: gatewayUrl,
      clientName: GATEWAY_CLIENT_NAMES.GATEWAY_CLIENT,
      clientDisplayName: "Telegram Cron Notifications",
      mode: GATEWAY_CLIENT_MODES.BACKEND,
      scopes: ["operator.admin"],
      onEvent: (evt) => this.handleGatewayEvent(evt),
      onHelloOk: () => {
        logDebug("telegram cron notifications: connected to gateway");
      },
      onConnectError: (err) => {
        logError(`telegram cron notifications: connect error: ${err.message}`);
      },
      onClose: (code, reason) => {
        logDebug(`telegram cron notifications: gateway closed: ${code} ${reason}`);
      },
    });

    this.gatewayClient.start();
  }

  async stop(): Promise<void> {
    if (!this.started) {
      return;
    }
    this.started = false;
    this.gatewayClient?.stop();
    this.gatewayClient = null;
  }

  private handleGatewayEvent(evt: EventFrame) {
    if (evt.event !== "cron") {
      return;
    }
    const parsed = parseCronMutationEvent(evt.payload);
    if (!parsed || !this.shouldHandle(parsed)) {
      return;
    }
    void this.handleCronMutation(parsed).catch((err) => {
      logError(`telegram cron notifications: failed to deliver: ${String(err)}`);
    });
  }

  private async handleCronMutation(event: CronMutationEvent) {
    const approvers = this.opts.config?.approvers ?? [];
    const message = buildCronMutationMessage(event);
    for (const approver of approvers) {
      const chatId = Number.parseInt(String(approver), 10);
      if (!Number.isFinite(chatId)) {
        continue;
      }
      await this.opts.bot.api.sendMessage(chatId, message, {
        parse_mode: "HTML",
      });
    }
  }
}

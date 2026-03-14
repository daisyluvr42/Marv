import { html } from "lit";
import { t, i18n, type Locale } from "../../i18n/index.js";
import { formatRelativeTimestamp, formatDurationHuman } from "../format.js";
import type { GatewayHelloOk } from "../gateway.js";
import type { OperationsSection, Tab } from "../navigation.js";
import { formatNextRun } from "../presenter.js";
import type { UiSettings } from "../storage.js";
import type {
  KnowledgeStatusSnapshot,
  MemoryStatusSnapshot,
  ProactiveStatusSnapshot,
} from "../types.js";

export type OverviewProps = {
  connected: boolean;
  hello: GatewayHelloOk | null;
  settings: UiSettings;
  password: string;
  lastError: string | null;
  trustedDeviceActive: boolean;
  presenceCount: number;
  sessionsCount: number | null;
  cronEnabled: boolean | null;
  cronNext: number | null;
  lastChannelsRefresh: number | null;
  dashboardLoading: boolean;
  dashboardError: string | null;
  memoryStats: MemoryStatusSnapshot | null;
  knowledgeStatus: KnowledgeStatusSnapshot | null;
  proactiveStatus: ProactiveStatusSnapshot | null;
  onSettingsChange: (next: UiSettings) => void;
  onPasswordChange: (next: string) => void;
  onSessionKeyChange: (next: string) => void;
  onConnect: () => void;
  onForgetDevice: () => void;
  onRefresh: () => void;
  onOpenOperationsSection: (section: OperationsSection) => void;
  onOpenTab: (tab: Tab) => void;
};

export function renderOverview(props: OverviewProps) {
  const formatTimestamp = (value: number | null | undefined) =>
    value ? formatRelativeTimestamp(value) : t("common.na");
  const formatBoolean = (value: boolean | null | undefined) =>
    value == null ? t("common.na") : value ? t("common.enabled") : t("common.disabled");
  const formatList = (values: string[] | undefined) =>
    values && values.length > 0 ? values.join(", ") : t("common.na");
  const tierSummary = props.memoryStats
    ? ["P0", "P1", "P2", "P3"]
        .map(
          (tier) =>
            `${tier} ${props.memoryStats?.tiers[tier as keyof typeof props.memoryStats.tiers] ?? 0}`,
        )
        .join(" · ")
    : t("common.na");
  const snapshot = props.hello?.snapshot as
    | {
        uptimeMs?: number;
        policy?: { tickIntervalMs?: number };
        authMode?: "none" | "token" | "password" | "trusted-proxy";
      }
    | undefined;
  const uptime = snapshot?.uptimeMs ? formatDurationHuman(snapshot.uptimeMs) : t("common.na");
  const tick = snapshot?.policy?.tickIntervalMs
    ? `${snapshot.policy.tickIntervalMs}ms`
    : t("common.na");
  const authMode = snapshot?.authMode;
  const isTrustedProxy = authMode === "trusted-proxy";
  const deviceStatus = props.trustedDeviceActive
    ? t("overview.access.trustedDevice")
    : t("overview.access.bootstrapOnly");

  const authHint = (() => {
    if (props.connected || !props.lastError) {
      return null;
    }
    const lower = props.lastError.toLowerCase();
    const authFailed = lower.includes("unauthorized") || lower.includes("connect failed");
    if (!authFailed) {
      return null;
    }
    const hasToken = Boolean(props.settings.token.trim());
    const hasPassword = Boolean(props.password.trim());
    if (!hasToken && !hasPassword) {
      return html`
        <div class="muted" style="margin-top: 8px">
          ${t("overview.auth.required")}
          <div style="margin-top: 6px">
            <span class="mono">marv dashboard --no-open</span> → tokenized URL<br />
            <span class="mono">marv doctor --generate-gateway-token</span> → set token
          </div>
          <div style="margin-top: 6px">
            <a
              class="session-link"
              href="#/web/dashboard"
              target="_blank"
              rel="noreferrer"
              title="Control UI auth docs (opens in new tab)"
              >Docs: Control UI auth</a
            >
          </div>
        </div>
      `;
    }
    return html`
      <div class="muted" style="margin-top: 8px">
        ${t("overview.auth.failed", { command: "marv dashboard --no-open" })}
        <div style="margin-top: 6px">
          <a
            class="session-link"
            href="#/web/dashboard"
            target="_blank"
            rel="noreferrer"
            title="Control UI auth docs (opens in new tab)"
            >Docs: Control UI auth</a
          >
        </div>
      </div>
    `;
  })();

  const insecureContextHint = (() => {
    if (props.connected || !props.lastError) {
      return null;
    }
    const isSecureContext = typeof window !== "undefined" ? window.isSecureContext : true;
    if (isSecureContext) {
      return null;
    }
    const lower = props.lastError.toLowerCase();
    if (!lower.includes("secure context") && !lower.includes("device identity required")) {
      return null;
    }
    return html`
      <div class="muted" style="margin-top: 8px">
        ${t("overview.insecure.hint", { url: "http://127.0.0.1:18789" })}
        <div style="margin-top: 6px">
          ${t("overview.insecure.stayHttp", { config: "gateway.controlUi.allowInsecureAuth: true" })}
        </div>
        <div style="margin-top: 6px">
          <a
            class="session-link"
            href="#/gateway/tailscale"
            target="_blank"
            rel="noreferrer"
            title="Tailscale Serve docs (opens in new tab)"
            >Docs: Tailscale Serve</a
          >
          <span class="muted"> · </span>
          <a
            class="session-link"
            href="#/web/control-ui#insecure-http"
            target="_blank"
            rel="noreferrer"
            title="Insecure HTTP docs (opens in new tab)"
            >Docs: Insecure HTTP</a
          >
        </div>
      </div>
    `;
  })();

  const currentLocale = i18n.getLocale();
  const attentionText = props.lastError ?? t("overview.attention.healthy");
  const healthLabel = props.connected ? t("common.ok") : t("common.offline");
  const quickLinks = [
    {
      label: t("overview.quickLinks.sessions"),
      action: () => props.onOpenOperationsSection("sessions"),
    },
    {
      label: t("overview.quickLinks.usage"),
      action: () => props.onOpenOperationsSection("usage"),
    },
    {
      label: t("overview.quickLinks.cron"),
      action: () => props.onOpenOperationsSection("cron"),
    },
    {
      label: t("overview.quickLinks.channels"),
      action: () => props.onOpenTab("channels"),
    },
    {
      label: t("overview.quickLinks.agents"),
      action: () => props.onOpenTab("agents"),
    },
    {
      label: t("overview.quickLinks.workspace"),
      action: () => props.onOpenTab("workspace"),
    },
    {
      label: t("overview.quickLinks.settings"),
      action: () => props.onOpenTab("settings"),
    },
    {
      label: t("overview.quickLinks.chat"),
      action: () => props.onOpenTab("chat"),
    },
  ];

  return html`
    <section class="grid grid-cols-3">
      <div class="card">
        <div class="card-title">${t("overview.summary.title")}</div>
        <div class="card-sub">${t("overview.summary.subtitle")}</div>
        <div class="stat-grid" style="margin-top: 16px;">
          <div class="stat">
            <div class="stat-label">${t("overview.summary.status")}</div>
            <div class="stat-value ${props.connected ? "ok" : "warn"}">${healthLabel}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.snapshot.uptime")}</div>
            <div class="stat-value">${uptime}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.summary.sessions")}</div>
            <div class="stat-value">${props.sessionsCount ?? t("common.na")}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.summary.instances")}</div>
            <div class="stat-value">${props.presenceCount}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.summary.cron")}</div>
            <div class="stat-value">${formatBoolean(props.cronEnabled)}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.summary.nextWake")}</div>
            <div class="stat-value">${formatNextRun(props.cronNext)}</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">${t("overview.attention.title")}</div>
        <div class="card-sub">${t("overview.attention.subtitle")}</div>
        <div class="callout ${props.lastError ? "danger" : ""}" style="margin-top: 16px;">
          ${attentionText}
          ${authHint ?? ""}
          ${insecureContextHint ?? ""}
        </div>
        <div class="row" style="margin-top: 14px; flex-wrap: wrap;">
          <button class="btn btn--sm" @click=${() => props.onOpenTab("channels")}>
            ${t("overview.attention.channels")}
          </button>
          <button class="btn btn--sm" @click=${() => props.onOpenOperationsSection("logs")}>
            ${t("overview.attention.logs")}
          </button>
          <button class="btn btn--sm" @click=${() => props.onOpenTab("settings")}>
            ${t("overview.attention.config")}
          </button>
          <button class="btn btn--sm" @click=${() => props.onOpenTab("chat")}>
            ${t("overview.attention.chat")}
          </button>
        </div>
      </div>

      <div class="card">
        <div class="card-title">${t("overview.access.title")}</div>
        <div class="card-sub">${t("overview.access.subtitle")}</div>
        <div class="form-grid" style="margin-top: 16px;">
          <label class="field">
            <span>${t("overview.access.wsUrl")}</span>
            <input
              .value=${props.settings.gatewayUrl}
              @input=${(e: Event) => {
                const v = (e.target as HTMLInputElement).value;
                props.onSettingsChange({ ...props.settings, gatewayUrl: v });
              }}
              placeholder="ws://100.x.y.z:18789"
            />
          </label>
          ${
            isTrustedProxy
              ? ""
              : html`
                <label class="field">
                  <span>${t("overview.access.token")}</span>
                  <input
                    .value=${props.settings.token}
                    @input=${(e: Event) => {
                      const v = (e.target as HTMLInputElement).value;
                      props.onSettingsChange({ ...props.settings, token: v });
                    }}
                    placeholder="MARV_GATEWAY_TOKEN"
                  />
                </label>
                <label class="field">
                  <span>${t("overview.access.password")}</span>
                  <input
                    type="password"
                    .value=${props.password}
                    @input=${(e: Event) => {
                      const v = (e.target as HTMLInputElement).value;
                      props.onPasswordChange(v);
                    }}
                    placeholder="system or shared password"
                  />
                </label>
              `
          }
          <label class="field">
            <span>${t("overview.access.sessionKey")}</span>
            <input
              .value=${props.settings.sessionKey}
              @input=${(e: Event) => {
                const v = (e.target as HTMLInputElement).value;
                props.onSessionKeyChange(v);
              }}
            />
          </label>
          <label class="field">
            <span>${t("overview.access.language")}</span>
            <select
              .value=${currentLocale}
              @change=${(e: Event) => {
                const v = (e.target as HTMLSelectElement).value as Locale;
                void i18n.setLocale(v);
                props.onSettingsChange({ ...props.settings, locale: v });
              }}
            >
              <option value="en">${t("languages.en")}</option>
              <option value="zh-CN">${t("languages.zhCN")}</option>
              <option value="zh-TW">${t("languages.zhTW")}</option>
              <option value="pt-BR">${t("languages.ptBR")}</option>
            </select>
          </label>
        </div>
        <div class="row" style="margin-top: 14px;">
          <button class="btn" @click=${() => props.onConnect()}>${t("common.connect")}</button>
          <button class="btn" @click=${() => props.onRefresh()}>${t("common.refresh")}</button>
          <button class="btn" @click=${() => props.onForgetDevice()}>
            ${t("overview.access.forgetDevice")}
          </button>
          <span class="muted">${
            isTrustedProxy ? t("overview.access.trustedProxy") : t("overview.access.connectHint")
          }</span>
        </div>
        <div class="muted" style="margin-top: 10px">${deviceStatus}</div>
      </div>
    </section>

    <section class="grid grid-cols-4" style="margin-top: 18px;">
      <div class="card stat-card">
        <div class="stat-label">${t("overview.stats.instances")}</div>
        <div class="stat-value">${props.presenceCount}</div>
        <div class="muted">${t("overview.stats.instancesHint")}</div>
      </div>
      <div class="card stat-card">
        <div class="stat-label">${t("overview.stats.sessions")}</div>
        <div class="stat-value">${props.sessionsCount ?? t("common.na")}</div>
        <div class="muted">${t("overview.stats.sessionsHint")}</div>
      </div>
      <div class="card stat-card">
        <div class="stat-label">${t("overview.stats.cron")}</div>
        <div class="stat-value">
          ${props.cronEnabled == null ? t("common.na") : props.cronEnabled ? t("common.enabled") : t("common.disabled")}
        </div>
        <div class="muted">${t("overview.stats.cronNext", { time: formatNextRun(props.cronNext) })}</div>
      </div>
      <div class="card stat-card">
        <div class="stat-label">${t("overview.snapshot.lastChannelsRefresh")}</div>
        <div class="stat-value">
          ${props.lastChannelsRefresh ? formatRelativeTimestamp(props.lastChannelsRefresh) : t("common.na")}
        </div>
        <div class="muted">${t("overview.snapshot.channelsHint")}</div>
      </div>
    </section>

    ${
      props.dashboardError
        ? html`
            <section class="callout danger" style="margin-top: 18px;">
              ${props.dashboardError}
            </section>
          `
        : ""
    }

    <section class="grid grid-cols-3" style="margin-top: 18px;">
      <div class="card">
        <div class="card-title">${t("overview.advanced.memoryTitle")}</div>
        <div class="card-sub">${t("overview.advanced.memorySubtitle")}</div>
        <div class="stat-grid" style="margin-top: 16px;">
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.totalItems")}</div>
            <div class="stat-value">${props.memoryStats?.totalItems ?? t("common.na")}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.archiveEvents")}</div>
            <div class="stat-value">${props.memoryStats?.archiveEvents ?? t("common.na")}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.autoRecall")}</div>
            <div class="stat-value">${formatBoolean(props.memoryStats?.autoRecallEnabled)}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.runtimeIngest")}</div>
            <div class="stat-value">${formatBoolean(props.memoryStats?.runtimeIngestEnabled)}</div>
          </div>
        </div>
        <div class="muted" style="margin-top: 12px">
          ${t("overview.advanced.tiers")}: ${tierSummary}
        </div>
        <div class="muted" style="margin-top: 8px">
          ${t("overview.advanced.backend")}: ${props.memoryStats?.backend ?? t("common.na")}
          <span class="muted"> · </span>
          ${t("overview.advanced.citations")}: ${props.memoryStats?.citations ?? t("common.na")}
        </div>
      </div>

      <div class="card">
        <div class="card-title">${t("overview.advanced.knowledgeTitle")}</div>
        <div class="card-sub">${t("overview.advanced.knowledgeSubtitle")}</div>
        <div class="stat-grid" style="margin-top: 16px;">
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.status")}</div>
            <div class="stat-value">${formatBoolean(props.knowledgeStatus?.enabled)}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.vaults")}</div>
            <div class="stat-value">${props.knowledgeStatus?.vaultCount ?? t("common.na")}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.files")}</div>
            <div class="stat-value">${props.knowledgeStatus?.totalFiles ?? t("common.na")}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.chunks")}</div>
            <div class="stat-value">${props.knowledgeStatus?.totalChunks ?? t("common.na")}</div>
          </div>
        </div>
        <div class="muted" style="margin-top: 12px">
          ${t("overview.advanced.lastScan")}: ${formatTimestamp(props.knowledgeStatus?.lastScanAt)}
        </div>
        <div class="muted" style="margin-top: 8px">
          ${t("overview.advanced.sync")}: search ${formatBoolean(props.knowledgeStatus?.autoSyncOnSearch)}
          <span class="muted"> · </span>
          boot ${formatBoolean(props.knowledgeStatus?.autoSyncOnBoot)}
        </div>
      </div>

      <div class="card">
        <div class="card-title">${t("overview.advanced.proactiveTitle")}</div>
        <div class="card-sub">${t("overview.advanced.proactiveSubtitle")}</div>
        <div class="stat-grid" style="margin-top: 16px;">
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.status")}</div>
            <div class="stat-value">${formatBoolean(props.proactiveStatus?.enabled)}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.pending")}</div>
            <div class="stat-value">${props.proactiveStatus?.pendingEntries ?? t("common.na")}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.urgent")}</div>
            <div class="stat-value">${props.proactiveStatus?.urgentEntries ?? t("common.na")}</div>
          </div>
          <div class="stat">
            <div class="stat-label">${t("overview.advanced.lastFlush")}</div>
            <div class="stat-value">${formatTimestamp(props.proactiveStatus?.lastFlushAt)}</div>
          </div>
        </div>
        <div class="muted" style="margin-top: 12px">
          ${t("overview.advanced.digests")}: ${formatList(props.proactiveStatus?.digestTimes)}
        </div>
        <div class="muted" style="margin-top: 8px">
          ${t("overview.advanced.delivery")}: ${props.proactiveStatus?.delivery.channel ?? t("common.na")}
          ${props.proactiveStatus?.delivery.to ? ` -> ${props.proactiveStatus.delivery.to}` : ""}
        </div>
        ${
          props.dashboardLoading
            ? html`<div class="muted" style="margin-top: 8px">${t("overview.advanced.loading")}</div>`
            : ""
        }
      </div>
    </section>

    <section class="card" style="margin-top: 18px;">
      <div class="row" style="justify-content: space-between; align-items: flex-start;">
        <div>
          <div class="card-title">${t("overview.quickLinks.title")}</div>
          <div class="card-sub">${t("overview.quickLinks.subtitle")}</div>
        </div>
        <div class="muted">${t("overview.snapshot.tickInterval")}: ${tick}</div>
      </div>
      <div class="section-rail__items" style="margin-top: 14px;">
        ${quickLinks.map(
          (item) => html`
            <button class="section-chip" @click=${item.action}>
              <span class="section-chip__title">${item.label}</span>
            </button>
          `,
        )}
      </div>
    </section>

    <section class="card" style="margin-top: 18px;">
      <div class="card-title">${t("overview.notes.title")}</div>
      <div class="card-sub">${t("overview.notes.subtitle")}</div>
      <div class="note-grid" style="margin-top: 14px;">
        <div>
          <div class="note-title">${t("overview.notes.tailscaleTitle")}</div>
          <div class="muted">
            ${t("overview.notes.tailscaleText")}
          </div>
        </div>
        <div>
          <div class="note-title">${t("overview.notes.sessionTitle")}</div>
          <div class="muted">${t("overview.notes.sessionText")}</div>
        </div>
        <div>
          <div class="note-title">${t("overview.notes.cronTitle")}</div>
          <div class="muted">${t("overview.notes.cronText")}</div>
        </div>
      </div>
    </section>
  `;
}

import { html, nothing, svg } from "lit";
import { formatDurationCompact } from "../../../../src/infra/format-time/format-duration.js";
import { clampText, formatRelativeTimestamp } from "../format.js";
import type { SessionUsageTimeSeries, SessionsUsageEntry, SessionsUsageResult } from "../types.js";
import { formatCost, formatTokens } from "./usage-metrics.js";
import type { SessionLogEntry } from "./usage.js";

export type ProjectsProps = {
  loading: boolean;
  error: string | null;
  result: SessionsUsageResult | null;
  query: string;
  selectedKey: string | null;
  timeSeries: SessionUsageTimeSeries | null;
  timeSeriesLoading: boolean;
  logs: SessionLogEntry[] | null;
  logsLoading: boolean;
  onRefresh: () => void;
  onQueryChange: (value: string) => void;
  onSelectSession: (sessionKey: string) => void;
};

function sessionTitle(session: SessionsUsageEntry): string {
  return session.label?.trim() || session.origin?.label?.trim() || session.key;
}

function renderBadges(session: SessionsUsageEntry) {
  const badges = [
    session.agentId ? `agent:${session.agentId}` : null,
    session.channel ? `channel:${session.channel}` : null,
    session.modelProvider ? `provider:${session.modelProvider}` : null,
    session.model ? `model:${session.model}` : null,
  ].filter((value): value is string => Boolean(value));
  if (badges.length === 0) {
    return nothing;
  }
  return html`
    <div class="usage-badges" style="margin-top: 8px;">
      ${badges.map((badge) => html`<span class="usage-badge">${badge}</span>`)}
    </div>
  `;
}

function renderSparkline(timeSeries: SessionUsageTimeSeries | null) {
  const points = timeSeries?.points ?? [];
  if (points.length < 2) {
    return html`
      <div
        class="card"
        style="
          display: flex;
          align-items: center;
          justify-content: center;
          height: 120px;
          border-style: dashed;
        "
      >
        <div class="muted">No timeline data yet.</div>
      </div>
    `;
  }
  const values = points.map((point) => point.totalTokens);
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = Math.max(max - min, 1);
  const coordinates = points.map((point, index) => {
    const x = (index / Math.max(points.length - 1, 1)) * 100;
    const y = 36 - ((point.totalTokens - min) / range) * 32;
    return `${x},${y}`;
  });
  return svg`
    <div class="card" style="padding: 16px;">
      <svg viewBox="0 0 100 40" style="width: 100%; height: 72px;">
        <polyline
          fill="none"
          stroke="var(--accent, #ef4444)"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          points=${coordinates.join(" ")}
        />
      </svg>
    </div>
  `;
}

export function renderProjects(props: ProjectsProps) {
  const sessions = (props.result?.sessions ?? []).filter((session) => {
    if (!props.query.trim()) {
      return true;
    }
    const needle = props.query.trim().toLowerCase();
    const haystack =
      `${session.key}\n${session.label ?? ""}\n${session.origin?.label ?? ""}\n${session.agentId ?? ""}\n${session.channel ?? ""}`.toLowerCase();
    return haystack.includes(needle);
  });
  const selected =
    sessions.find((session) => session.key === props.selectedKey) ??
    props.result?.sessions.find((session) => session.key === props.selectedKey) ??
    null;
  const usage = selected?.usage ?? null;
  return html`
    <section class="grid grid-master-detail" style="gap: 20px;">
      <div style="display: flex; flex-direction: column; gap: 16px; min-height: 0;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <div style="font-size: 1.125rem; font-weight: 600;">Projects</div>
            <div class="muted" style="font-size: 0.875rem;">Recent session activity</div>
          </div>
          <button class="btn-secondary" style="padding: 4px 12px; font-size: 0.875rem;" @click=${props.onRefresh}>Refresh</button>
        </div>
        <div>
          <input
            class="input"
            style="width: 100%;"
            .value=${props.query}
            @input=${(event: Event) =>
              props.onQueryChange((event.target as HTMLInputElement).value)}
            placeholder="Search sessions…"
          />
          <div style="display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;">
            <span class="pill" style="cursor: pointer;">Recent</span>
            <span class="pill" style="cursor: pointer;">Tokens</span>
            <span class="pill" style="cursor: pointer;">Cost</span>
            <span class="pill" style="cursor: pointer;">Errors</span>
          </div>
        </div>
        ${props.error ? html`<div class="pill danger">${props.error}</div>` : nothing}
        ${
          props.loading
            ? html`
                <div class="muted">Loading project activity…</div>
              `
            : nothing
        }
        <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 8px;">
          ${
            sessions.length === 0
              ? html`
                  <div class="muted">No sessions matched this range.</div>
                `
              : sessions.map((session) => {
                  const usage = session.usage;
                  const active = session.key === props.selectedKey;
                  return html`
                    <button
                      type="button"
                      class="card"
                      style="text-align: left; padding: 12px 16px; transition: all 0.2s; border: 1px solid ${active ? "var(--accent, #ef4444)" : "var(--border, rgba(255,255,255,0.12))"}; background: ${active ? "rgba(239, 68, 68, 0.05)" : "var(--surface-sunken)"}; cursor: pointer;"
                      @click=${() => props.onSelectSession(session.key)}
                    >
                      <div style="display: flex; justify-content: space-between; gap: 12px; align-items: baseline;">
                        <div style="font-weight: 600; font-size: 0.9375rem;">${sessionTitle(session)}</div>
                        <div class="muted" style="font-size: 0.75rem;">${usage?.lastActivity ? formatRelativeTimestamp(usage.lastActivity).replace(" ago", "") : "n/a"}</div>
                      </div>
                      ${renderBadges(session)}
                    </button>
                  `;
                })
          }
        </div>
      </div>
      
      <div style="position: sticky; top: 16px; display: flex; flex-direction: column; gap: 24px;">
        ${
          !selected || !usage
            ? html`
                <div
                  class="card"
                  style="
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 300px;
                    text-align: center;
                    border-style: dashed;
                  "
                >
                  <div style="font-weight: 600; margin-bottom: 4px">No Session Selected</div>
                  <div class="muted" style="max-width: 250px">
                    Pick a session from the list to inspect its usage summary, timeline, and recent logs.
                  </div>
                </div>
              `
            : html`
                <div>
                  <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 4px;">${sessionTitle(selected)}</div>
                  <div class="muted mono" style="font-size: 0.75rem; user-select: text;">${selected.key}</div>
                </div>
                
                <div class="card" style="padding: 20px;">
                  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Messages</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px;">${usage.messageCounts?.total ?? 0}</div>
                    </div>
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Tools</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px;">${usage.toolUsage?.totalCalls ?? 0}</div>
                    </div>
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase; ${usage.messageCounts?.errors && usage.messageCounts.errors > 0 ? "color: var(--danger);" : ""}">Errors</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px; ${usage.messageCounts?.errors && usage.messageCounts.errors > 0 ? "color: var(--danger);" : ""}">${usage.messageCounts?.errors ?? 0}</div>
                    </div>
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Duration</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px;">${formatDurationCompact(usage.durationMs, { spaced: true }) ?? "n/a"}</div>
                    </div>
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Tokens</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px;">${formatTokens(usage.totalTokens)}</div>
                    </div>
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Cost</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px;">${formatCost(usage.totalCost)}</div>
                    </div>
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Active Days</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px;">${usage.activityDates?.length ?? 0}</div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <div style="font-weight: 600; margin-bottom: 12px; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);">Activity Timeline</div>
                  ${
                    props.timeSeriesLoading
                      ? html`
                          <div
                            class="card"
                            style="
                              display: flex;
                              align-items: center;
                              justify-content: center;
                              height: 120px;
                              border-style: dashed;
                            "
                          >
                            <div class="muted">Loading timeline…</div>
                          </div>
                        `
                      : renderSparkline(props.timeSeries)
                  }
                </div>
                
                <div>
                  <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px;">
                    <div style="font-weight: 600; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);">Recent Logs</div>
                  </div>
                  ${
                    props.logsLoading
                      ? html`
                          <div class="muted">Loading logs…</div>
                        `
                      : !props.logs || props.logs.length === 0
                        ? html`
                            <div class="card" style="padding: 24px; text-align: center; border-style: dashed">
                              <div class="muted">No session logs captured yet.</div>
                            </div>
                          `
                        : html`
                            <div style="display: flex; flex-direction: column; gap: 8px;">
                              ${props.logs
                                .slice(-12)
                                .toReversed()
                                .map(
                                  (entry) => html`
                                  <div class="card" style="padding: 12px 14px; background: var(--surface-sunken);">
                                    <div style="display: flex; justify-content: space-between; gap: 12px; align-items: center;">
                                      <div class="pill ${String(entry.role) === "error" ? "danger" : ""}" style="font-size: 0.7rem;">${entry.role}</div>
                                      <div class="muted" style="font-size: 0.75rem;">${entry.timestamp ? formatRelativeTimestamp(entry.timestamp) : "n/a"}</div>
                                    </div>
                                    <div style="margin-top: 8px; font-size: 0.875rem; font-family: var(--font-mono); white-space: pre-wrap; word-break: break-all; opacity: 0.9;">${clampText(entry.content, 220)}</div>
                                  </div>
                                `,
                                )}
                            </div>
                          `
                  }
                </div>
              `
        }
      </div>
    </section>
  `;
}

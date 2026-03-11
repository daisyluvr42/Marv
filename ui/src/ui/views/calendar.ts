import { html, nothing } from "lit";
import { formatRelativeTimestamp } from "../format.js";
import type { WorkspaceCalendarSnapshot } from "../workspace-types.js";
import { formatCost, formatTokens } from "./usage-metrics.js";

export type CalendarProps = {
  loading: boolean;
  error: string | null;
  snapshot: WorkspaceCalendarSnapshot | null;
  selectedDay: string | null;
  onRefresh: () => void;
  onSelectDay: (date: string) => void;
};

function weekdayLabel(date: string): string {
  return new Date(`${date}T00:00:00`).toLocaleDateString(undefined, { weekday: "short" });
}

export function renderCalendar(props: CalendarProps) {
  const days = props.snapshot?.days ?? [];
  const selectedDay =
    days.find((day) => day.date === props.selectedDay) ?? days[days.length - 1] ?? null;
  const maxTokens = Math.max(...days.map((day) => day.tokens), 1);
  return html`
    <section class="grid" style="grid-template-columns: minmax(320px, 5fr) minmax(400px, 5fr); align-items: start; gap: 24px;">
      <div style="display: flex; flex-direction: column; gap: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <div style="font-size: 1.125rem; font-weight: 600;">Calendar</div>
            <div class="muted" style="font-size: 0.875rem;">Daily intensity heatmap and anomalies</div>
          </div>
          <button class="btn-secondary" style="padding: 4px 12px; font-size: 0.875rem;" @click=${props.onRefresh}>Refresh</button>
        </div>
        
        ${props.error ? html`<div class="pill danger">${props.error}</div>` : nothing}
        ${
          props.loading
            ? html`
                <div class="muted">Loading recent days…</div>
              `
            : nothing
        }
        
        <div
          style="margin-top: 8px; display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 6px;"
        >
          ${days.map((day) => {
            const intensity = Math.min(day.tokens / maxTokens, 1);
            const selected = day.date === selectedDay?.date;
            const hasCronError = day.cronRuns.some(
              (run) =>
                run.status.toLowerCase().includes("fail") ||
                run.status.toLowerCase().includes("error"),
            );
            const background =
              day.tokens > 0 || day.cronRuns.length > 0 ? `var(--surface-sunken)` : "transparent";

            let heatmapStyle = `background: ${background};`;
            if (day.tokens > 0) {
              heatmapStyle = `background: rgba(239, 68, 68, ${0.1 + intensity * 0.7});`;
            }

            return html`
              <button
                type="button"
                class="card"
                style="padding: 8px; text-align: left; transition: all 0.2s; position: relative; border: 1px solid ${selected ? "var(--accent, #ef4444)" : "var(--border, rgba(255,255,255,0.08))"}; ${heatmapStyle} cursor: pointer;"
                @click=${() => props.onSelectDay(day.date)}
              >
                ${
                  hasCronError
                    ? html`
                        <div
                          style="
                            position: absolute;
                            top: -4px;
                            right: -4px;
                            width: 8px;
                            height: 8px;
                            background: var(--danger);
                            border-radius: 50%;
                            border: 2px solid var(--bg);
                          "
                        ></div>
                      `
                    : nothing
                }
                <div style="font-weight: 600; font-size: 0.875rem;">${day.date.slice(8)}</div>
                <div class="muted" style="font-size: 0.65rem; text-transform: uppercase;">${weekdayLabel(day.date)}</div>
              </button>
            `;
          })}
        </div>
      </div>
      
      <div style="position: sticky; top: 16px; display: flex; flex-direction: column; gap: 24px;">
        ${
          !selectedDay
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
                  <div style="font-weight: 600; margin-bottom: 4px">No Day Selected</div>
                  <div class="muted" style="max-width: 250px">
                    Select an activity block from the calendar heatmap.
                  </div>
                </div>
              `
            : html`
                <div>
                  <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 4px;">${selectedDay.date}</div>
                  <div class="muted" style="font-size: 0.875rem;">${weekdayLabel(selectedDay.date)}</div>
                </div>
                
                <div class="card" style="padding: 20px;">
                  <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Tokens</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px;">${formatTokens(selectedDay.tokens)}</div>
                    </div>
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Cost</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px;">${formatCost(selectedDay.cost)}</div>
                    </div>
                    <div>
                        <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Messages</div>
                        <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px;">${selectedDay.messages}</div>
                    </div>
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Sessions</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px;">${selectedDay.sessionCount}</div>
                    </div>
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Tools</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px;">${selectedDay.toolCalls}</div>
                    </div>
                    <div>
                      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase; ${selectedDay.errors > 0 ? "color: var(--danger);" : ""}">Errors</div>
                      <div style="font-size: 1.125rem; font-weight: 600; margin-top: 4px; ${selectedDay.errors > 0 ? "color: var(--danger);" : ""}">${selectedDay.errors}</div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <div style="font-weight: 600; margin-bottom: 12px; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);">Top Sessions</div>
                  ${
                    selectedDay.topSessions.length === 0
                      ? html`
                          <div class="card" style="padding: 24px; text-align: center; border-style: dashed">
                            <div class="muted">No session activity recorded for this day.</div>
                          </div>
                        `
                      : html`
                          <div style="display: flex; flex-direction: column; gap: 8px;">
                            ${selectedDay.topSessions.map(
                              (session) => html`
                                <div class="card" style="padding: 12px 14px; background: var(--surface-sunken);">
                                  <div style="display: flex; justify-content: space-between; gap: 12px; align-items: baseline;">
                                    <div>
                                      <div style="font-weight: 600; font-size: 0.9375rem;">${session.label ?? session.key}</div>
                                      <div class="muted mono" style="font-size: 0.75rem;">${session.key}</div>
                                    </div>
                                    <div class="muted" style="font-size: 0.75rem;">${session.lastActivity ? formatRelativeTimestamp(session.lastActivity).replace(" ago", "") : "n/a"}</div>
                                  </div>
                                  <div style="margin-top: 8px; font-size: 0.875rem; display: flex; gap: 8px; opacity: 0.9;">
                                    <span class="pill" style="font-size: 0.75rem;">Tokens: ${formatTokens(session.tokens)}</span>
                                    <span class="pill" style="font-size: 0.75rem;">Cost: ${formatCost(session.cost)}</span>
                                  </div>
                                </div>
                              `,
                            )}
                          </div>
                        `
                  }
                </div>
                
                <div>
                  <div style="font-weight: 600; margin-bottom: 12px; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);">Cron Runs</div>
                  ${
                    selectedDay.cronRuns.length === 0
                      ? html`
                          <div class="card" style="padding: 24px; text-align: center; border-style: dashed">
                            <div class="muted">No cron runs landed on this day.</div>
                          </div>
                        `
                      : html`
                          <div style="display: flex; flex-direction: column; gap: 8px;">
                            ${selectedDay.cronRuns
                              .toSorted((a, b) => b.ts - a.ts)
                              .slice(0, 8)
                              .map((run) => {
                                const isError =
                                  run.status.toLowerCase().includes("fail") ||
                                  run.status.toLowerCase().includes("error");
                                return html`
                                    <div class="card" style="padding: 12px 14px; background: var(--surface-sunken); border-left: ${isError ? "3px solid var(--danger)" : "1px solid var(--border)"};">
                                      <div style="display: flex; justify-content: space-between; gap: 12px; align-items: baseline;">
                                        <div style="font-weight: 600; font-size: 0.9375rem;">${run.jobName ?? run.jobId}</div>
                                        <div class="muted" style="font-size: 0.75rem;">${formatRelativeTimestamp(run.ts).replace(" ago", "")}</div>
                                      </div>
                                      <div style="margin-top: 6px; display: flex; align-items: center; gap: 8px;">
                                        <span class="pill ${isError ? "danger" : ""}" style="font-size: 0.7rem;">${run.status}</span>
                                        ${run.agentId ? html`<span class="muted" style="font-size: 0.75rem;">Agent: ${run.agentId}</span>` : nothing}
                                      </div>
                                      ${run.summary ? html`<div class="muted" style="margin-top: 8px; font-size: 0.875rem;">${run.summary}</div>` : nothing}
                                    </div>
                                  `;
                              })}
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

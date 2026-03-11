import { html, nothing } from "lit";
import { formatRelativeTimestamp } from "../format.js";
import type { WorkspaceSummarySnapshot } from "../workspace-types.js";
import { formatCost, formatTokens } from "./usage-metrics.js";

type WorkspaceSummaryProps = {
  loading: boolean;
  error: string | null;
  summary: WorkspaceSummarySnapshot | null;
};

function renderStripItem(label: string, value: string, sub?: string, isDanger?: boolean) {
  return html`
    <div style="display: flex; flex-direction: column; gap: 2px;">
      <div class="muted" style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;">${label}</div>
      <div style="font-size: 1rem; font-weight: 600; font-variant-numeric: tabular-nums; ${isDanger ? "color: var(--danger);" : ""}">${value}</div>
      ${sub ? html`<div class="muted" style="font-size: 0.75rem;">${sub}</div>` : nothing}
    </div>
  `;
}

export function renderWorkspaceSummary(props: WorkspaceSummaryProps) {
  return html`
    <section class="card" style="margin-bottom: 24px; padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; gap: 24px; flex-wrap: wrap;">
      <div style="display: flex; align-items: center; gap: 12px; min-width: max-content;">
        <div style="font-weight: 600;">Workspace Context</div>
        ${
          props.loading
            ? html`
                <span class="pill">Refreshing…</span>
              `
            : nothing
        }
        ${props.error ? html`<span class="pill danger">${props.error}</span>` : nothing}
      </div>

      <div style="display: flex; align-items: flex-start; gap: 32px; flex: 1; justify-content: flex-end; flex-wrap: wrap;">
        ${renderStripItem(
          "Sessions (7d)",
          props.summary ? String(props.summary.sessionsTouched) : "0",
          props.summary ? `${props.summary.activeDays} active days` : undefined,
        )}
        ${renderStripItem(
          "Proactive Buffer",
          props.summary ? `${props.summary.pendingProactive} pending` : "0 pending",
          props.summary && props.summary.urgentProactive > 0
            ? `${props.summary.urgentProactive} urgent`
            : undefined,
          props.summary ? props.summary.urgentProactive > 0 : false,
        )}
        ${renderStripItem(
          "Next Cron Wake",
          props.summary?.nextWakeAtMs ? formatRelativeTimestamp(props.summary.nextWakeAtMs) : "n/a",
          props.summary && props.summary.failingJobs > 0
            ? `${props.summary.failingJobs} jobs failing`
            : undefined,
          props.summary ? props.summary.failingJobs > 0 : false,
        )}
        ${renderStripItem(
          "Total Usage (7d)",
          props.summary ? formatTokens(props.summary.totalTokens) : "0 tokens",
          props.summary ? formatCost(props.summary.totalCost) : undefined,
        )}
      </div>
    </section>
  `;
}

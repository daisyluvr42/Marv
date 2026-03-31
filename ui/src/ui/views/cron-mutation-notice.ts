import { html, nothing } from "lit";
import type { AppViewState } from "../app-view-state.js";

function actionTitle(action: "added" | "updated" | "removed"): string {
  if (action === "added") {
    return "Cron job added";
  }
  if (action === "updated") {
    return "Cron job updated";
  }
  return "Cron job removed";
}

function renderMeta(label: string, value?: string | null) {
  if (!value) {
    return nothing;
  }
  return html`<span>${label}: <span class="mono">${value}</span></span>`;
}

export function renderCronMutationNotices(state: AppViewState) {
  if (state.cronMutationNotices.length === 0) {
    return nothing;
  }
  return html`
    <div class="cron-mutation-toast-stack" aria-live="polite">
      ${state.cronMutationNotices.map(
        (entry) => html`
          <div class="cron-mutation-toast">
            <div class="cron-mutation-toast__title">${actionTitle(entry.action)}</div>
            <div class="cron-mutation-toast__body">${entry.jobName || entry.jobId}</div>
            <div class="cron-mutation-toast__meta">
              ${renderMeta("Job", entry.jobId)}
              ${renderMeta("Agent", entry.agentId)}
              ${renderMeta("Target", entry.sessionTarget)}
              ${renderMeta("Delivery", entry.deliveryMode)}
              ${
                typeof entry.nextRunAtMs === "number"
                  ? renderMeta("Next", new Date(entry.nextRunAtMs).toISOString())
                  : nothing
              }
            </div>
          </div>
        `,
      )}
    </div>
  `;
}

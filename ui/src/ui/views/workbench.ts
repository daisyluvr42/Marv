import { html, nothing } from "lit";
import { formatRelativeTimestamp } from "../format.js";
import type {
  WorkspaceWorkbenchDeepLink,
  WorkspaceWorkbenchRow,
  WorkspaceWorkbenchSnapshot,
  WorkspaceWorkbenchStatus,
} from "../workspace-types.js";

export type WorkbenchProps = {
  loading: boolean;
  error: string | null;
  snapshot: WorkspaceWorkbenchSnapshot | null;
  onRefresh: () => void;
  onOpenDeepLink: (link: WorkspaceWorkbenchDeepLink) => void;
};

function statusTone(status: WorkspaceWorkbenchStatus): string {
  switch (status) {
    case "active":
      return "background: rgba(16, 185, 129, 0.12); color: #10b981;";
    case "blocked":
      return "background: rgba(239, 68, 68, 0.12); color: #ef4444;";
    case "queued":
      return "background: rgba(245, 158, 11, 0.12); color: #f59e0b;";
    case "paused":
      return "background: rgba(148, 163, 184, 0.12); color: #94a3b8;";
    case "completed":
      return "background: rgba(59, 130, 246, 0.12); color: #60a5fa;";
    case "archived":
      return "background: rgba(100, 116, 139, 0.12); color: #94a3b8;";
    default:
      return "";
  }
}

function renderRow(
  row: WorkspaceWorkbenchRow,
  onOpenDeepLink: (link: WorkspaceWorkbenchDeepLink) => void,
) {
  return html`
    <div class="list-item" style="align-items: flex-start;">
      <div class="list-main" style="gap: 10px;">
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
          <div class="list-title">${row.title}</div>
          <span class="pill" style="font-size: 0.72rem; text-transform: uppercase; ${statusTone(row.status)}"
            >${row.status}</span
          >
          <span class="pill" style="font-size: 0.72rem;">${row.source}</span>
        </div>
        ${
          row.summary
            ? html`<div class="list-sub" style="max-width: 56rem; line-height: 1.55;">${row.summary}</div>`
            : nothing
        }
      </div>
      <div class="list-meta" style="align-items: flex-end; gap: 8px;">
        <div class="muted" style="font-size: 0.75rem;">
          ${formatRelativeTimestamp(Date.parse(row.updatedAt))}
        </div>
        ${
          row.deepLink
            ? html`
              <button class="btn-secondary" style="padding: 4px 10px; font-size: 0.8rem;" @click=${() => onOpenDeepLink(row.deepLink!)}>
                Open
              </button>
            `
            : nothing
        }
      </div>
    </div>
  `;
}

function renderSection(params: {
  title: string;
  subtitle: string;
  rows: WorkspaceWorkbenchRow[];
  empty: string;
  onOpenDeepLink: (link: WorkspaceWorkbenchDeepLink) => void;
}) {
  return html`
    <section class="card" style="padding: 18px;">
      <div class="card-title">${params.title}</div>
      <div class="card-sub">${params.subtitle}</div>
      ${
        params.rows.length === 0
          ? html`<div class="muted" style="margin-top: 14px;">${params.empty}</div>`
          : html`
            <div class="list" style="margin-top: 14px;">
              ${params.rows.map((row) => renderRow(row, params.onOpenDeepLink))}
            </div>
          `
      }
    </section>
  `;
}

export function renderWorkbench(props: WorkbenchProps) {
  const rows = props.snapshot?.rows ?? [];
  const activeTaskContexts = rows.filter(
    (row) =>
      row.source === "task-context" &&
      (row.status === "active" || row.status === "paused" || row.status === "blocked"),
  );
  const proactiveQueue = rows.filter(
    (row) =>
      row.source !== "task-context" && row.status !== "completed" && row.status !== "archived",
  );
  const historicalRows = rows.filter(
    (row) => row.status === "completed" || row.status === "archived",
  );

  return html`
    <section style="display: flex; flex-direction: column; gap: 20px;">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap;">
        <div>
          <div style="font-size: 1.125rem; font-weight: 600;">Work</div>
          <div class="muted" style="font-size: 0.875rem;">Current task context, proactive queue, and recent completions.</div>
        </div>
        <button class="btn-secondary" style="padding: 4px 12px; font-size: 0.875rem;" @click=${props.onRefresh}>
          Refresh
        </button>
      </div>

      ${props.error ? html`<div class="pill danger">${props.error}</div>` : nothing}

      <section class="card" style="padding: 16px; display: flex; gap: 22px; flex-wrap: wrap; align-items: flex-start;">
        <div>
          <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Active</div>
          <div style="font-size: 1.25rem; font-weight: 600;">${props.snapshot?.counts.active ?? 0}</div>
        </div>
        <div>
          <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Queued</div>
          <div style="font-size: 1.25rem; font-weight: 600;">${props.snapshot?.counts.queued ?? 0}</div>
        </div>
        <div>
          <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Blocked / Paused</div>
          <div style="font-size: 1.25rem; font-weight: 600;">${(props.snapshot?.counts.blocked ?? 0) + (props.snapshot?.counts.paused ?? 0)}</div>
        </div>
        <div>
          <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Deliverables</div>
          <div style="font-size: 1.25rem; font-weight: 600;">${props.snapshot?.deliverableSummary.completed ?? 0} / ${props.snapshot?.deliverableSummary.total ?? 0}</div>
        </div>
        <div>
          <div class="muted" style="font-size: 0.75rem; text-transform: uppercase;">Fetched</div>
          <div style="font-size: 1rem; font-weight: 600;">
            ${props.snapshot ? formatRelativeTimestamp(Date.parse(props.snapshot.fetchedAt)) : "n/a"}
          </div>
        </div>
      </section>

      ${
        props.loading && !props.snapshot
          ? html`
              <div class="muted">Loading workbench…</div>
            `
          : nothing
      }

      ${renderSection({
        title: "Task Context",
        subtitle: "Live task threads and paused work carried by the agent.",
        rows: activeTaskContexts,
        empty: "No active task contexts right now.",
        onOpenDeepLink: props.onOpenDeepLink,
      })}

      ${renderSection({
        title: "Proactive Queue",
        subtitle: "Goals and queued/running proactive tasks.",
        rows: proactiveQueue,
        empty: "No proactive items are queued right now.",
        onOpenDeepLink: props.onOpenDeepLink,
      })}

      <details class="card" style="padding: 18px;" ?open=${historicalRows.length > 0 && historicalRows.length <= 3}>
        <summary style="cursor: pointer; font-weight: 600;">Completed / Archived (${historicalRows.length})</summary>
        ${
          historicalRows.length === 0
            ? html`
                <div class="muted" style="margin-top: 14px">No completed items yet.</div>
              `
            : html`
              <div class="list" style="margin-top: 14px;">
                ${historicalRows.map((row) => renderRow(row, props.onOpenDeepLink))}
              </div>
            `
        }
      </details>
    </section>
  `;
}

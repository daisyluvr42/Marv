import { html, nothing } from "lit";
import { clampText, formatMs, formatRelativeTimestamp } from "../format.js";
import type {
  WorkspaceDocumentEntry,
  WorkspaceDocumentsListResult,
  WorkspaceDocumentsReadResult,
} from "../workspace-types.js";

export type DocumentsProps = {
  loading: boolean;
  error: string | null;
  query: string;
  result: WorkspaceDocumentsListResult | null;
  selectedRootId: string | null;
  selectedPath: string | null;
  readLoading: boolean;
  readError: string | null;
  readResult: WorkspaceDocumentsReadResult | null;
  onRefresh: () => void;
  onQueryChange: (value: string) => void;
  onSelect: (rootId: string, relativePath: string) => void;
};

function renderDocumentRow(
  entry: WorkspaceDocumentEntry,
  active: boolean,
  onSelect: (rootId: string, relativePath: string) => void,
) {
  return html`
    <button
      type="button"
      class="card"
      style="text-align: left; padding: 12px 14px; transition: all 0.2s; border: 1px solid ${active ? "var(--accent, #ef4444)" : "var(--border, rgba(255,255,255,0.12))"}; background: ${active ? "rgba(239, 68, 68, 0.05)" : "var(--surface-sunken)"}; cursor: pointer;"
      @click=${() => onSelect(entry.rootId, entry.relativePath)}
    >
      <div style="display: flex; justify-content: space-between; gap: 12px; align-items: baseline;">
        <div style="font-weight: 600; font-size: 0.9375rem; word-break: break-all;">${entry.name}</div>
        <div class="muted" style="font-size: 0.75rem; white-space: nowrap;">${formatRelativeTimestamp(entry.mtimeMs).replace(" ago", "")}</div>
      </div>
      <div class="muted mono" style="font-size: 0.75rem; margin-top: 4px; word-break: break-all;">${entry.relativePath}</div>
      
      <div style="display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap;">
        ${entry.agentIds.map((agent) => html`<span class="pill" style="font-size: 0.7rem;">ag:${agent}</span>`)}
        <span class="pill" style="font-size: 0.7rem; text-transform: uppercase;">${entry.category}</span>
        <span class="pill" style="font-size: 0.7rem; font-family: var(--font-mono);">${entry.extension || "text"}</span>
      </div>
      ${entry.preview ? html`<div class="muted" style="margin-top: 12px; font-size: 0.8125rem; line-height: 1.4; opacity: 0.9;">${clampText(entry.preview, 120)}</div>` : nothing}
    </button>
  `;
}

export function renderDocuments(props: DocumentsProps) {
  const items = props.result?.items ?? [];
  return html`
    <section class="grid grid-master-detail" style="gap: 24px;">
      <div style="display: flex; flex-direction: column; gap: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <div style="font-size: 1.125rem; font-weight: 600;">Documents</div>
            <div class="muted" style="font-size: 0.875rem;">Workspace files across agents</div>
          </div>
          <button class="btn-secondary" style="padding: 4px 12px; font-size: 0.875rem;" @click=${props.onRefresh}>Refresh</button>
        </div>
        
        <div>
          <input
            class="input"
            style="width: 100%;"
            .value=${props.query}
            @input=${(event: Event) => props.onQueryChange((event.target as HTMLInputElement).value)}
            placeholder="Search documents by path or preview…"
          />
        </div>
        
        ${props.error ? html`<div class="pill danger">${props.error}</div>` : nothing}
        ${
          props.loading
            ? html`
                <div class="muted">Loading workspace files…</div>
              `
            : nothing
        }
        
        <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 8px;">
          ${
            items.length === 0
              ? html`
                  <div class="muted">No documents matched the current filter.</div>
                `
              : items.map((entry) =>
                  renderDocumentRow(
                    entry,
                    entry.rootId === props.selectedRootId &&
                      entry.relativePath === props.selectedPath,
                    props.onSelect,
                  ),
                )
          }
        </div>
      </div>
      
      <div style="position: sticky; top: 16px; display: flex; flex-direction: column; gap: 16px;">
        ${
          props.readError
            ? html`<div class="pill danger">${props.readError}</div>`
            : props.readLoading
              ? html`
                  <div
                    class="card"
                    style="
                      display: flex;
                      align-items: center;
                      justify-content: center;
                      height: 300px;
                      border-style: dashed;
                    "
                  >
                    <div class="muted">Loading document preview…</div>
                  </div>
                `
              : !props.readResult
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
                      <div style="font-weight: 600; margin-bottom: 4px">No Document Selected</div>
                      <div class="muted" style="max-width: 250px">
                        Pick a file from the list to inspect its content and metadata.
                      </div>
                    </div>
                  `
                : html`
                    <div class="card" style="padding: 24px; background: var(--surface-sunken); display: flex; flex-direction: column; height: calc(100vh - 120px); max-height: 800px;">
                      
                      <div style="border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 16px;">
                        <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 6px; word-break: break-word;">${props.readResult.name}</div>
                        <div class="muted mono" style="font-size: 0.8125rem; user-select: text; word-break: break-all; margin-bottom: 12px;">${props.readResult.relativePath}</div>
                        
                        <div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center; font-size: 0.8125rem;">
                          <div style="display: flex; align-items: center; gap: 6px;">
                            <span class="muted">Agents:</span>
                            <span style="font-weight: 500;">${props.readResult.agentIds.join(", ")}</span>
                          </div>
                          <div style="display: flex; align-items: center; gap: 6px;">
                            <span class="muted">Size:</span>
                            <span style="font-weight: 500;">${(props.readResult.size / 1024).toFixed(1)} KB</span>
                          </div>
                          <div style="display: flex; align-items: center; gap: 6px;">
                            <span class="muted">Modified:</span>
                            <span style="font-weight: 500;">${formatMs(props.readResult.mtimeMs)}</span>
                          </div>
                        </div>
                      </div>

                      ${
                        props.readResult.truncated
                          ? html`
                              <div class="pill warning" style="margin-bottom: 16px; align-self: flex-start">
                                Preview truncated to keep the UI responsive.
                              </div>
                            `
                          : nothing
                      }
                      
                      <div style="flex: 1; overflow: auto; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 16px;">
                        <pre style="margin: 0; font-family: var(--font-mono); font-size: 0.875rem; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; color: var(--text);">${props.readResult.content}</pre>
                      </div>
                      
                    </div>
                  `
        }
      </div>
    </section>
  `;
}

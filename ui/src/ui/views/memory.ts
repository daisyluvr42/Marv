import { html, nothing } from "lit";
import { clampText, formatRelativeTimestamp } from "../format.js";
import type {
  WorkspaceMemoryItem,
  WorkspaceMemoryListResult,
  WorkspaceMemorySearchResult,
} from "../workspace-types.js";

export type MemoryProps = {
  loading: boolean;
  error: string | null;
  query: string;
  list: WorkspaceMemoryListResult | null;
  search: WorkspaceMemorySearchResult | null;
  onRefresh: () => void;
  onQueryChange: (value: string) => void;
  onSearch: () => void;
  onClear: () => void;
};

function renderMemoryCard(
  item: WorkspaceMemoryItem,
  extra?: { score?: number; references?: number },
) {
  return html`
    <div class="card" style="padding: 16px; background: var(--surface-sunken); border: 1px solid var(--border);">
      <div style="display: flex; justify-content: space-between; gap: 16px; align-items: flex-start;">
        <div style="font-weight: 600; font-size: 1rem; line-height: 1.4;">${item.summary?.trim() || clampText(item.content, 72)}</div>
        <div class="muted" style="font-size: 0.75rem; white-space: nowrap;">${formatRelativeTimestamp(item.createdAt)}</div>
      </div>
      <div style="display: flex; gap: 6px; margin-top: 12px; flex-wrap: wrap;">
        <span class="pill" style="font-size: 0.7rem; text-transform: uppercase;">${item.recordKind}</span>
        <span class="pill" style="font-size: 0.7rem;">${item.source}</span>
        <span class="pill" style="font-size: 0.7rem; font-family: var(--font-mono);">${item.scopeType}:${item.scopeId}</span>
        <span class="pill" style="font-size: 0.7rem;">${item.recordKind}</span>
        ${extra?.score != null ? html`<span class="pill" style="font-size: 0.7rem; background: rgba(239, 68, 68, 0.1); color: var(--accent, #ef4444); border-color: rgba(239, 68, 68, 0.2);">Match: ${(extra.score * 100).toFixed(0)}%</span>` : nothing}
        ${extra?.references != null && extra.references > 0 ? html`<span class="pill" style="font-size: 0.7rem;">${extra.references} refs</span>` : nothing}
      </div>
      <div class="muted" style="margin-top: 16px; font-size: 0.9375rem; line-height: 1.6; white-space: pre-wrap; opacity: 0.85;">${clampText(item.content, 240)}</div>
    </div>
  `;
}

export function renderMemory(props: MemoryProps) {
  const isSearchState = props.search != null;
  const searchItems = props.search?.items ?? [];
  const listItems = props.list?.items ?? [];
  const items = isSearchState ? searchItems : listItems;

  const renderedItems = isSearchState
    ? searchItems.map((item) =>
        renderMemoryCard(item, {
          score: item.score,
          references: item.references.length,
        }),
      )
    : items.map((item) => renderMemoryCard(item));

  return html`
    <section style="display: flex; flex-direction: column; gap: 24px; max-width: 1000px; margin: 0 auto; width: 100%;">
      <div style="display: flex; flex-direction: column; gap: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <div style="font-size: 1.125rem; font-weight: 600;">Memory</div>
            <div class="muted" style="font-size: 0.875rem;">Vector knowledge and conversation recall</div>
          </div>
          <button class="btn-secondary" style="padding: 4px 12px; font-size: 0.875rem;" @click=${props.onRefresh}>Refresh</button>
        </div>
        
        <div style="display: flex; gap: 12px; align-items: center;">
          <input
            class="input"
            style="flex: 1;"
            .value=${props.query}
            @input=${(event: Event) => props.onQueryChange((event.target as HTMLInputElement).value)}
            @keydown=${(event: KeyboardEvent) => {
              if (event.key === "Enter") {
                props.onSearch();
              }
            }}
            placeholder="Search memory content, entities, or summaries…"
          />
          <button class="btn-primary" style="padding: 8px 16px;" @click=${props.onSearch}>Search</button>
          ${isSearchState || props.query ? html`<button class="btn-secondary" style="padding: 8px 16px;" @click=${props.onClear}>Clear</button>` : nothing}
        </div>
        
        ${
          isSearchState && props.search?.scopes?.length
            ? html`
              <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <span class="muted" style="font-size: 0.875rem;">Searching in:</span>
                ${props.search.scopes.map((scope) => html`<span class="pill" style="font-size: 0.75rem;">${scope.scopeType}:${scope.scopeId}</span>`)}
              </div>
            `
            : nothing
        }
        
        ${props.error ? html`<div class="pill danger">${props.error}</div>` : nothing}
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px;">
          <div style="font-weight: 600; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);">
            ${isSearchState ? `Search Results (${items.length})` : "Recent Memories"}
          </div>
        </div>
        
        ${
          props.loading
            ? html`
                <div class="muted">Loading memory items…</div>
              `
            : items.length === 0
              ? html`
                  <div class="card" style="padding: 40px 20px; text-align: center; border-style: dashed;">
                    <div style="font-weight: 600; margin-bottom: 8px;">${isSearchState ? "No Matches Found" : "No Memories Stored"}</div>
                    <div class="muted" style="max-width: 300px; margin: 0 auto;">
                      ${
                        isSearchState
                          ? "Try adjusting your search terms or clearing the filter to browse recent items."
                          : "Memory items will appear here as the system learns and stores context."
                      }
                    </div>
                  </div>
                `
              : html`
                  <div style="display: grid; gap: 16px;">
                    ${renderedItems}
                  </div>
                `
        }
      </div>
    </section>
  `;
}

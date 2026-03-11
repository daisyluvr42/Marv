import type { GatewayBrowserClient } from "../gateway.js";
import type { WorkspaceMemoryListResult, WorkspaceMemorySearchResult } from "../workspace-types.js";

export type WorkspaceMemoryState = {
  client: GatewayBrowserClient | null;
  connected: boolean;
  sessionKey: string;
  workspaceMemoryLoading: boolean;
  workspaceMemoryError: string | null;
  workspaceMemoryQuery: string;
  workspaceMemoryList: WorkspaceMemoryListResult | null;
  workspaceMemorySearch: WorkspaceMemorySearchResult | null;
};

export async function loadWorkspaceMemory(state: WorkspaceMemoryState) {
  if (!state.client || !state.connected || state.workspaceMemoryLoading) {
    return;
  }
  state.workspaceMemoryLoading = true;
  state.workspaceMemoryError = null;
  try {
    const list = await state.client.request<WorkspaceMemoryListResult>("memory.list", {
      sessionKey: state.sessionKey,
      limit: 120,
    });
    state.workspaceMemoryList = list;
    if (!state.workspaceMemoryQuery.trim()) {
      state.workspaceMemorySearch = null;
    }
  } catch (error) {
    state.workspaceMemoryError = String(error);
  } finally {
    state.workspaceMemoryLoading = false;
  }
}

export async function searchWorkspaceMemory(state: WorkspaceMemoryState) {
  if (!state.workspaceMemoryQuery.trim()) {
    state.workspaceMemorySearch = null;
    if (!state.workspaceMemoryList) {
      await loadWorkspaceMemory(state);
    }
    return;
  }
  if (!state.client || !state.connected || state.workspaceMemoryLoading) {
    return;
  }
  state.workspaceMemoryLoading = true;
  state.workspaceMemoryError = null;
  try {
    const result = await state.client.request<WorkspaceMemorySearchResult>("memory.search", {
      sessionKey: state.sessionKey,
      query: state.workspaceMemoryQuery.trim(),
      topK: 25,
    });
    state.workspaceMemorySearch = result;
  } catch (error) {
    state.workspaceMemoryError = String(error);
  } finally {
    state.workspaceMemoryLoading = false;
  }
}

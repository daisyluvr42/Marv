import type { GatewayBrowserClient } from "../gateway.js";
import type {
  WorkspaceDocumentsListResult,
  WorkspaceDocumentsReadResult,
} from "../workspace-types.js";

export type WorkspaceDocumentsState = {
  client: GatewayBrowserClient | null;
  connected: boolean;
  workspaceDocumentsLoading: boolean;
  workspaceDocumentsError: string | null;
  workspaceDocumentsQuery: string;
  workspaceDocumentsResult: WorkspaceDocumentsListResult | null;
  workspaceDocumentsSelectedRootId: string | null;
  workspaceDocumentsSelectedPath: string | null;
  workspaceDocumentReadLoading: boolean;
  workspaceDocumentReadError: string | null;
  workspaceDocumentReadResult: WorkspaceDocumentsReadResult | null;
};

export async function loadWorkspaceDocuments(state: WorkspaceDocumentsState) {
  if (!state.client || !state.connected || state.workspaceDocumentsLoading) {
    return;
  }
  state.workspaceDocumentsLoading = true;
  state.workspaceDocumentsError = null;
  try {
    const result = await state.client.request<WorkspaceDocumentsListResult>("documents.list", {
      query: state.workspaceDocumentsQuery.trim() || undefined,
      limit: 200,
      sort: "recent",
    });
    state.workspaceDocumentsResult = result;
    const selectedExists = result.items.some(
      (item) =>
        item.rootId === state.workspaceDocumentsSelectedRootId &&
        item.relativePath === state.workspaceDocumentsSelectedPath,
    );
    const nextSelection = selectedExists ? null : result.items[0];
    if (result.items.length === 0) {
      state.workspaceDocumentsSelectedRootId = null;
      state.workspaceDocumentsSelectedPath = null;
      state.workspaceDocumentReadResult = null;
    } else if (nextSelection) {
      await selectWorkspaceDocument(state, nextSelection.rootId, nextSelection.relativePath);
    } else if (
      !state.workspaceDocumentReadResult &&
      state.workspaceDocumentsSelectedRootId &&
      state.workspaceDocumentsSelectedPath
    ) {
      await selectWorkspaceDocument(
        state,
        state.workspaceDocumentsSelectedRootId,
        state.workspaceDocumentsSelectedPath,
      );
    }
  } catch (error) {
    state.workspaceDocumentsError = String(error);
  } finally {
    state.workspaceDocumentsLoading = false;
  }
}

export async function selectWorkspaceDocument(
  state: Pick<
    WorkspaceDocumentsState,
    | "client"
    | "connected"
    | "workspaceDocumentsSelectedRootId"
    | "workspaceDocumentsSelectedPath"
    | "workspaceDocumentReadLoading"
    | "workspaceDocumentReadError"
    | "workspaceDocumentReadResult"
  >,
  rootId: string,
  relativePath: string,
) {
  if (!state.client || !state.connected || !rootId || !relativePath) {
    return;
  }
  state.workspaceDocumentsSelectedRootId = rootId;
  state.workspaceDocumentsSelectedPath = relativePath;
  state.workspaceDocumentReadLoading = true;
  state.workspaceDocumentReadError = null;
  try {
    state.workspaceDocumentReadResult = await state.client.request<WorkspaceDocumentsReadResult>(
      "documents.read",
      {
        rootId,
        relativePath,
      },
    );
  } catch (error) {
    state.workspaceDocumentReadError = String(error);
  } finally {
    state.workspaceDocumentReadLoading = false;
  }
}

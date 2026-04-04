import type { GatewayBrowserClient } from "../gateway.js";
import type { WorkspaceWorkbenchSnapshot } from "../workspace-types.js";

export const WORKBENCH_POLL_INTERVAL_MS = 30_000;

export type WorkspaceWorkbenchState = {
  client: Pick<GatewayBrowserClient, "request"> | null;
  connected: boolean;
  workspaceWorkbenchLoading: boolean;
  workspaceWorkbenchError: string | null;
  workspaceWorkbench: WorkspaceWorkbenchSnapshot | null;
};

export async function loadWorkbench(
  state: WorkspaceWorkbenchState,
  overrides?: {
    agentId?: string;
  },
) {
  if (!state.client || !state.connected || state.workspaceWorkbenchLoading) {
    return;
  }
  state.workspaceWorkbenchLoading = true;
  state.workspaceWorkbenchError = null;
  try {
    const params = overrides?.agentId?.trim() ? { agentId: overrides.agentId.trim() } : {};
    state.workspaceWorkbench = await state.client.request<WorkspaceWorkbenchSnapshot>(
      "workbench.status",
      params,
    );
  } catch (error) {
    state.workspaceWorkbenchError = String(error);
  } finally {
    state.workspaceWorkbenchLoading = false;
  }
}

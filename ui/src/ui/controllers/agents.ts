import type { GatewayBrowserClient } from "../gateway.js";
import type { AgentsListResult, GatewayModelChoice } from "../types.js";

export type AgentsState = {
  client: GatewayBrowserClient | null;
  connected: boolean;
  agentsLoading: boolean;
  agentsError: string | null;
  agentsList: AgentsListResult | null;
  agentModelsLoading: boolean;
  agentModels: GatewayModelChoice[];
  agentsSelectedId: string | null;
};

export async function loadAgents(state: AgentsState) {
  if (!state.client || !state.connected) {
    return;
  }
  if (state.agentsLoading) {
    return;
  }
  state.agentsLoading = true;
  state.agentModelsLoading = true;
  state.agentsError = null;
  try {
    const [agentsResult, modelsResult] = await Promise.allSettled([
      state.client.request<AgentsListResult>("agents.list", {}),
      state.client.request<{ models?: GatewayModelChoice[] }>("models.list", {}),
    ]);

    if (agentsResult.status === "fulfilled" && agentsResult.value) {
      const res = agentsResult.value;
      state.agentsList = res;
      const selected = state.agentsSelectedId;
      const known = res.agents.some((entry) => entry.id === selected);
      if (!selected || !known) {
        state.agentsSelectedId = res.defaultId ?? res.agents[0]?.id ?? null;
      }
    } else if (agentsResult.status === "rejected") {
      state.agentsError = String(agentsResult.reason);
    }

    if (modelsResult.status === "fulfilled") {
      const payload = modelsResult.value;
      state.agentModels = Array.isArray(payload?.models) ? payload.models : [];
    }
  } finally {
    state.agentsLoading = false;
    state.agentModelsLoading = false;
  }
}

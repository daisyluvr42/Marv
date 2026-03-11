import type { GatewayBrowserClient } from "../gateway.js";
import type {
  KnowledgeStatusSnapshot,
  MemoryStatusSnapshot,
  ProactiveStatusSnapshot,
} from "../types.js";

export type DashboardState = {
  client: GatewayBrowserClient | null;
  connected: boolean;
  dashboardLoading: boolean;
  dashboardError: string | null;
  memoryStats: MemoryStatusSnapshot | null;
  knowledgeStatus: KnowledgeStatusSnapshot | null;
  proactiveStatus: ProactiveStatusSnapshot | null;
};

export async function loadDashboard(
  state: DashboardState,
  overrides?: {
    agentId?: string;
  },
) {
  if (!state.client || !state.connected) {
    return;
  }
  if (state.dashboardLoading) {
    return;
  }
  state.dashboardLoading = true;
  state.dashboardError = null;
  const params = overrides?.agentId?.trim() ? { agentId: overrides.agentId.trim() } : {};
  const [memoryResult, knowledgeResult, proactiveResult] = await Promise.allSettled([
    state.client.request<MemoryStatusSnapshot>("memory.stats", params),
    state.client.request<KnowledgeStatusSnapshot>("knowledge.status", params),
    state.client.request<ProactiveStatusSnapshot>("proactive.buffer", params),
  ]);
  if (memoryResult.status === "fulfilled") {
    state.memoryStats = memoryResult.value;
  }
  if (knowledgeResult.status === "fulfilled") {
    state.knowledgeStatus = knowledgeResult.value;
  }
  if (proactiveResult.status === "fulfilled") {
    state.proactiveStatus = proactiveResult.value;
  }
  const errors = [memoryResult, knowledgeResult, proactiveResult]
    .filter((result): result is PromiseRejectedResult => result.status === "rejected")
    .map((result) => String(result.reason));
  state.dashboardError = errors.length > 0 ? errors.join(" | ") : null;
  state.dashboardLoading = false;
}

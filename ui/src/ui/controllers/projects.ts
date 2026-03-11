import type { GatewayBrowserClient } from "../gateway.js";
import type { SessionsUsageResult, SessionUsageTimeSeries } from "../types.js";
import type { SessionLogEntry } from "../views/usage.js";
import { getRecentDateRange } from "./workspace-date.js";

export type WorkspaceProjectsState = {
  client: GatewayBrowserClient | null;
  connected: boolean;
  workspaceProjectsLoading: boolean;
  workspaceProjectsError: string | null;
  workspaceProjectsResult: SessionsUsageResult | null;
  workspaceProjectsRangeStart: string;
  workspaceProjectsRangeEnd: string;
  workspaceProjectsQuery: string;
  workspaceProjectsSelectedKey: string | null;
  workspaceProjectTimeSeries: SessionUsageTimeSeries | null;
  workspaceProjectTimeSeriesLoading: boolean;
  workspaceProjectLogs: SessionLogEntry[] | null;
  workspaceProjectLogsLoading: boolean;
};

export async function loadWorkspaceProjects(state: WorkspaceProjectsState) {
  if (!state.client || !state.connected || state.workspaceProjectsLoading) {
    return;
  }
  state.workspaceProjectsLoading = true;
  state.workspaceProjectsError = null;
  try {
    if (!state.workspaceProjectsRangeStart || !state.workspaceProjectsRangeEnd) {
      const range = getRecentDateRange(30);
      state.workspaceProjectsRangeStart = range.startDate;
      state.workspaceProjectsRangeEnd = range.endDate;
    }
    const result = await state.client.request<SessionsUsageResult>("sessions.usage", {
      startDate: state.workspaceProjectsRangeStart,
      endDate: state.workspaceProjectsRangeEnd,
      limit: 250,
      includeContextWeight: true,
    });
    state.workspaceProjectsResult = result;
    const selectedExists = result.sessions.some(
      (session) => session.key === state.workspaceProjectsSelectedKey,
    );
    const nextSelectedKey = selectedExists
      ? state.workspaceProjectsSelectedKey
      : (result.sessions[0]?.key ?? null);
    state.workspaceProjectsSelectedKey = nextSelectedKey;
    if (nextSelectedKey) {
      await loadWorkspaceProjectDetails(state, nextSelectedKey);
    } else {
      state.workspaceProjectTimeSeries = null;
      state.workspaceProjectLogs = null;
    }
  } catch (error) {
    state.workspaceProjectsError = String(error);
  } finally {
    state.workspaceProjectsLoading = false;
  }
}

export async function selectWorkspaceProjectSession(
  state: WorkspaceProjectsState,
  sessionKey: string,
) {
  if (!sessionKey) {
    return;
  }
  state.workspaceProjectsSelectedKey = sessionKey;
  await loadWorkspaceProjectDetails(state, sessionKey);
}

export async function loadWorkspaceProjectDetails(
  state: Pick<
    WorkspaceProjectsState,
    | "client"
    | "connected"
    | "workspaceProjectTimeSeries"
    | "workspaceProjectTimeSeriesLoading"
    | "workspaceProjectLogs"
    | "workspaceProjectLogsLoading"
  >,
  sessionKey: string,
) {
  if (!state.client || !state.connected || !sessionKey) {
    return;
  }
  state.workspaceProjectTimeSeriesLoading = true;
  state.workspaceProjectLogsLoading = true;
  const [timeSeriesResult, logsResult] = await Promise.allSettled([
    state.client.request<SessionUsageTimeSeries>("sessions.usage.timeseries", {
      key: sessionKey,
    }),
    state.client.request<{ logs?: SessionLogEntry[] }>("sessions.usage.logs", {
      key: sessionKey,
      limit: 200,
    }),
  ]);
  state.workspaceProjectTimeSeries =
    timeSeriesResult.status === "fulfilled" ? timeSeriesResult.value : null;
  state.workspaceProjectLogs =
    logsResult.status === "fulfilled" && Array.isArray(logsResult.value.logs)
      ? logsResult.value.logs
      : null;
  state.workspaceProjectTimeSeriesLoading = false;
  state.workspaceProjectLogsLoading = false;
}

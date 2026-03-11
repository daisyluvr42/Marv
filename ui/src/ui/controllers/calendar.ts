import type { GatewayBrowserClient } from "../gateway.js";
import type {
  CronJob,
  CronRunLogEntry,
  SessionsUsageEntry,
  SessionsUsageResult,
} from "../types.js";
import type {
  WorkspaceCalendarDay,
  WorkspaceCalendarSnapshot,
  WorkspaceCalendarTopSession,
} from "../workspace-types.js";
import { dateKeyFromTimestamp, enumerateDateRange, getRecentDateRange } from "./workspace-date.js";

type SessionDayCounter = WorkspaceCalendarTopSession & { _key: string };

export type WorkspaceCalendarState = {
  client: GatewayBrowserClient | null;
  connected: boolean;
  workspaceCalendarLoading: boolean;
  workspaceCalendarError: string | null;
  workspaceCalendar: WorkspaceCalendarSnapshot | null;
  workspaceCalendarSelectedDay: string | null;
};

function buildDayMap(startDate: string, endDate: string) {
  return new Map<string, WorkspaceCalendarDay>(
    enumerateDateRange(startDate, endDate).map((date) => [
      date,
      {
        date,
        tokens: 0,
        cost: 0,
        messages: 0,
        toolCalls: 0,
        errors: 0,
        sessionCount: 0,
        topSessions: [],
        cronRuns: [],
      },
    ]),
  );
}

function mergeSessionBreakdowns(
  dayMap: Map<string, WorkspaceCalendarDay>,
  session: SessionsUsageEntry,
  countersByDate: Map<string, Map<string, SessionDayCounter>>,
) {
  const usage = session.usage;
  if (!usage) {
    return;
  }
  const breakdowns = usage.dailyBreakdown ?? [];
  const messageCountsByDate = new Map(
    (usage.dailyMessageCounts ?? []).map((entry) => [entry.date, entry]),
  );
  for (const breakdown of breakdowns) {
    const day = dayMap.get(breakdown.date);
    if (!day) {
      continue;
    }
    const messageCounts = messageCountsByDate.get(breakdown.date);
    const bySession = countersByDate.get(breakdown.date) ?? new Map<string, SessionDayCounter>();
    const existing = bySession.get(session.key) ?? {
      _key: session.key,
      key: session.key,
      label: session.label,
      agentId: session.agentId,
      tokens: 0,
      cost: 0,
      messages: 0,
      lastActivity: usage.lastActivity,
    };
    existing.tokens += breakdown.tokens;
    existing.cost += breakdown.cost;
    existing.messages += messageCounts?.total ?? 0;
    existing.lastActivity = usage.lastActivity ?? session.updatedAt;
    bySession.set(session.key, existing);
    countersByDate.set(breakdown.date, bySession);
  }
}

function buildCalendarSnapshot(params: {
  startDate: string;
  endDate: string;
  usage: SessionsUsageResult;
  cronJobs: CronJob[];
  cronRunsByJobId: Map<string, CronRunLogEntry[]>;
}): WorkspaceCalendarSnapshot {
  const dayMap = buildDayMap(params.startDate, params.endDate);
  for (const daily of params.usage.aggregates.daily) {
    const day = dayMap.get(daily.date);
    if (!day) {
      continue;
    }
    day.tokens = daily.tokens;
    day.cost = daily.cost;
    day.messages = daily.messages;
    day.toolCalls = daily.toolCalls;
    day.errors = daily.errors;
  }

  const countersByDate = new Map<string, Map<string, SessionDayCounter>>();
  for (const session of params.usage.sessions) {
    mergeSessionBreakdowns(dayMap, session, countersByDate);
  }
  for (const [date, counters] of countersByDate.entries()) {
    const day = dayMap.get(date);
    if (!day) {
      continue;
    }
    day.sessionCount = counters.size;
    day.topSessions = [...counters.values()]
      .toSorted((a, b) => b.tokens - a.tokens || b.cost - a.cost || b.messages - a.messages)
      .slice(0, 5)
      .map(({ _key, ...session }) => session);
  }

  for (const job of params.cronJobs) {
    const entries = params.cronRunsByJobId.get(job.id) ?? [];
    for (const entry of entries) {
      const date = dateKeyFromTimestamp(entry.ts);
      const day = dayMap.get(date);
      if (!day) {
        continue;
      }
      day.cronRuns.push({
        ...entry,
        jobName: job.name,
        agentId: job.agentId,
      });
    }
  }

  return {
    startDate: params.startDate,
    endDate: params.endDate,
    days: [...dayMap.values()],
  };
}

export async function loadWorkspaceCalendar(state: WorkspaceCalendarState) {
  if (!state.client || !state.connected || state.workspaceCalendarLoading) {
    return;
  }
  const client = state.client;
  state.workspaceCalendarLoading = true;
  state.workspaceCalendarError = null;
  try {
    const { startDate, endDate } = getRecentDateRange(35);
    const usage = await client.request<SessionsUsageResult>("sessions.usage", {
      startDate,
      endDate,
      limit: 1000,
    });
    const cronList = await client.request<{ jobs?: CronJob[] }>("cron.list", {
      includeDisabled: true,
    });
    const cronJobs = Array.isArray(cronList.jobs) ? cronList.jobs : [];
    const cronRunsByJobId = new Map<string, CronRunLogEntry[]>();
    const runResults = await Promise.allSettled(
      cronJobs.slice(0, 24).map(async (job) => {
        const result = await client.request<{ entries?: CronRunLogEntry[] }>("cron.runs", {
          id: job.id,
          limit: 25,
        });
        return {
          jobId: job.id,
          entries: Array.isArray(result?.entries) ? result.entries : [],
        };
      }),
    );
    for (const result of runResults) {
      if (result.status === "fulfilled") {
        cronRunsByJobId.set(result.value.jobId, result.value.entries);
      }
    }
    const snapshot = buildCalendarSnapshot({
      startDate,
      endDate,
      usage,
      cronJobs,
      cronRunsByJobId,
    });
    state.workspaceCalendar = snapshot;
    const hasSelectedDay = snapshot.days.some(
      (day) => day.date === state.workspaceCalendarSelectedDay,
    );
    const defaultDay =
      snapshot.days
        .toReversed()
        .find(
          (day) =>
            day.messages > 0 || day.tokens > 0 || day.cronRuns.length > 0 || day.sessionCount > 0,
        )?.date ?? snapshot.endDate;
    state.workspaceCalendarSelectedDay = hasSelectedDay
      ? state.workspaceCalendarSelectedDay
      : defaultDay;
  } catch (error) {
    state.workspaceCalendarError = String(error);
  } finally {
    state.workspaceCalendarLoading = false;
  }
}

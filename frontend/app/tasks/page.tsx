'use client';

import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';

type TaskStatusPayload = {
  id: string;
  conversation_id: string;
  status: string;
  created_at: number;
  updated_at: number;
  last_error: string | null;
  current_stage: string;
};

type StreamEvent = {
  event_id: string;
  type: string;
  ts: number;
  payload: Record<string, unknown>;
};

type AuditPayload = {
  summary: {
    event_count: number;
    tool_call_count: number;
    approval_count: number;
  };
  timeline: StreamEvent[];
  tool_calls: Array<{
    tool_call_id: string;
    tool: string;
    status: string;
    approval_id: string | null;
    error: string | null;
  }>;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';

export default function TasksPage() {
  const [taskIdInput, setTaskIdInput] = useState('');
  const [task, setTask] = useState<TaskStatusPayload | null>(null);
  const [audit, setAudit] = useState<AuditPayload | null>(null);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [following, setFollowing] = useState(false);
  const [error, setError] = useState('');
  const streamRef = useRef<EventSource | null>(null);

  const resolvedTaskId = task?.id ?? taskIdInput.trim();

  const eventTypeSummary = useMemo(() => {
    const counts = new Map<string, number>();
    for (const item of events) {
      counts.set(item.type, (counts.get(item.type) ?? 0) + 1);
    }
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  }, [events]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.close();
      }
    };
  }, []);

  const readError = async (response: Response): Promise<string> => {
    try {
      const payload = (await response.json()) as { detail?: string };
      if (typeof payload.detail === 'string' && payload.detail.trim()) {
        return payload.detail;
      }
    } catch {
      // ignore parse error
    }
    return `request failed: ${response.status}`;
  };

  const loadTask = async (taskId: string): Promise<TaskStatusPayload | null> => {
    const response = await fetch(`${API_BASE}/v1/agent/tasks/${encodeURIComponent(taskId)}`);
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    const payload = (await response.json()) as TaskStatusPayload;
    setTask(payload);
    return payload;
  };

  const loadAudit = async (taskId: string): Promise<void> => {
    const response = await fetch(`${API_BASE}/v1/audit/render`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ task_id: taskId })
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    const payload = (await response.json()) as AuditPayload;
    setAudit(payload);
  };

  const stopFollowing = () => {
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }
    setFollowing(false);
  };

  const startFollowing = (taskId: string) => {
    stopFollowing();
    setFollowing(true);
    const stream = new EventSource(`${API_BASE}/v1/agent/tasks/${encodeURIComponent(taskId)}/events`);
    streamRef.current = stream;

    stream.onmessage = (msg) => {
      try {
        const parsed = JSON.parse(msg.data) as StreamEvent;
        setEvents((prev) => [...prev, parsed]);
      } catch {
        // ignore invalid event payload
      }
    };

    stream.addEventListener('done', () => {
      stopFollowing();
      void loadTask(taskId).catch(() => undefined);
      void loadAudit(taskId).catch(() => undefined);
    });

    stream.onerror = () => {
      stopFollowing();
    };
  };

  const onLookup = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const taskId = taskIdInput.trim();
    if (!taskId) {
      return;
    }

    setError('');
    setLoading(true);
    setEvents([]);
    setAudit(null);
    try {
      await loadTask(taskId);
      await loadAudit(taskId);
    } catch (err) {
      setTask(null);
      setAudit(null);
      setError(err instanceof Error ? err.message : 'unknown error');
    } finally {
      setLoading(false);
    }
  };

  const refreshAll = async () => {
    const taskId = resolvedTaskId.trim();
    if (!taskId) {
      return;
    }
    setError('');
    setLoading(true);
    try {
      await loadTask(taskId);
      await loadAudit(taskId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="space-y-4">
      <header>
        <h1 className="text-2xl font-semibold">Tasks</h1>
        <p className="mt-2 text-sm text-muted">按 task_id 查看状态、审计轨迹和实时事件流。</p>
      </header>

      <form onSubmit={onLookup} className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-4 md:flex-row">
        <input
          value={taskIdInput}
          onChange={(e) => setTaskIdInput(e.target.value)}
          placeholder="输入 task_id"
          className="flex-1 rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text outline-none focus:border-primary"
        />
        <button type="submit" disabled={loading} className="rounded-md bg-primary px-4 py-2 text-sm text-white disabled:opacity-60">
          {loading ? 'Loading...' : 'Load Task'}
        </button>
      </form>

      {error ? <p className="text-sm text-danger">{error}</p> : null}

      {task ? (
        <div className="rounded-lg border border-border bg-surface p-4">
          <div className="grid grid-cols-1 gap-2 text-sm md:grid-cols-2">
            <p>
              <span className="text-muted">task_id:</span> <span className="font-mono">{task.id}</span>
            </p>
            <p>
              <span className="text-muted">conversation:</span> <span className="font-mono">{task.conversation_id}</span>
            </p>
            <p>
              <span className="text-muted">status:</span> {task.status}
            </p>
            <p>
              <span className="text-muted">stage:</span> {task.current_stage}
            </p>
            <p>
              <span className="text-muted">created_at:</span> {task.created_at}
            </p>
            <p>
              <span className="text-muted">updated_at:</span> {task.updated_at}
            </p>
          </div>
          {task.last_error ? <p className="mt-2 text-sm text-danger">last_error: {task.last_error}</p> : null}
          <div className="mt-3 flex flex-wrap gap-2">
            <button onClick={() => void refreshAll()} className="rounded-md border border-border px-3 py-1 text-sm text-text">
              Refresh
            </button>
            {!following ? (
              <button onClick={() => startFollowing(task.id)} className="rounded-md bg-primary/20 px-3 py-1 text-sm text-primary">
                Follow Events
              </button>
            ) : (
              <button onClick={stopFollowing} className="rounded-md bg-warning/20 px-3 py-1 text-sm text-warning">
                Stop Stream
              </button>
            )}
          </div>
        </div>
      ) : null}

      <div className="rounded-lg border border-border bg-surface p-4">
        <h2 className="text-sm font-medium">Live Events</h2>
        {following ? <p className="mt-1 text-xs text-primary">stream connected</p> : null}
        {eventTypeSummary.length > 0 ? (
          <div className="mt-2 flex flex-wrap gap-2 text-xs">
            {eventTypeSummary.map(([type, count]) => (
              <span key={type} className="rounded border border-border px-2 py-1 text-muted">
                {type}: {count}
              </span>
            ))}
          </div>
        ) : null}
        <div className="mt-3 max-h-80 space-y-2 overflow-auto">
          {events.map((eventItem) => (
            <div key={eventItem.event_id} className="rounded border border-border bg-surface2 p-2 text-xs">
              <p className="font-mono text-muted">{eventItem.type}</p>
              <pre className="mt-1 overflow-x-auto text-[11px]">{JSON.stringify(eventItem.payload, null, 2)}</pre>
            </div>
          ))}
          {events.length === 0 ? <p className="text-xs text-muted">No streamed events yet.</p> : null}
        </div>
      </div>

      <div className="rounded-lg border border-border bg-surface p-4">
        <h2 className="text-sm font-medium">Audit Snapshot</h2>
        {audit ? (
          <>
            <div className="mt-2 flex flex-wrap gap-3 text-xs text-muted">
              <span>events: {audit.summary.event_count}</span>
              <span>tool_calls: {audit.summary.tool_call_count}</span>
              <span>approvals: {audit.summary.approval_count}</span>
            </div>
            <div className="mt-3 max-h-72 space-y-2 overflow-auto">
              {audit.timeline.map((item) => (
                <div key={item.event_id} className="rounded border border-border bg-surface2 p-2 text-xs">
                  <p className="font-mono text-muted">{item.type}</p>
                  <pre className="mt-1 overflow-x-auto text-[11px]">{JSON.stringify(item.payload, null, 2)}</pre>
                </div>
              ))}
            </div>
            <div className="mt-3">
              <h3 className="text-xs font-medium text-text">Tool Calls</h3>
              <div className="mt-2 space-y-2">
                {audit.tool_calls.map((call) => (
                  <div key={call.tool_call_id} className="rounded border border-border bg-surface2 p-2 text-xs">
                    <p className="font-mono text-muted">{call.tool_call_id}</p>
                    <p className="mt-1">
                      tool={call.tool} status={call.status}
                    </p>
                    {call.approval_id ? <p className="text-muted">approval={call.approval_id}</p> : null}
                    {call.error ? <p className="text-danger">error={call.error}</p> : null}
                  </div>
                ))}
                {audit.tool_calls.length === 0 ? <p className="text-xs text-muted">No tool calls.</p> : null}
              </div>
            </div>
          </>
        ) : (
          <p className="mt-2 text-xs text-muted">Load a task first to view audit snapshot.</p>
        )}
      </div>
    </section>
  );
}


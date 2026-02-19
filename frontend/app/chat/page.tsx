'use client';

import { FormEvent, useMemo, useState } from 'react';

type StreamEvent = {
  event_id: string;
  type: string;
  ts: number;
  payload: Record<string, unknown>;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';

export default function ChatPage() {
  const [message, setMessage] = useState('');
  const [conversationId, setConversationId] = useState<string>('');
  const [taskId, setTaskId] = useState<string>('');
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string>('');

  const completionText = useMemo(() => {
    const completion = [...events].reverse().find((event) => event.type === 'CompletionEvent');
    return (completion?.payload?.response_text as string | undefined) ?? '';
  }, [events]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!message.trim() || isRunning) {
      return;
    }

    setError('');
    setEvents([]);
    setIsRunning(true);

    try {
      const response = await fetch(`${API_BASE}/v1/agent/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Actor-Role': 'owner',
          'X-Actor-Id': 'web-owner'
        },
        body: JSON.stringify({
          message,
          conversation_id: conversationId || undefined,
          channel: 'web'
        })
      });
      if (!response.ok) {
        throw new Error(`send failed: ${response.status}`);
      }

      const payload = (await response.json()) as { task_id: string; conversation_id: string };
      setTaskId(payload.task_id);
      setConversationId(payload.conversation_id);

      await new Promise<void>((resolve) => {
        const stream = new EventSource(`${API_BASE}/v1/agent/tasks/${payload.task_id}/events`);
        stream.onmessage = (msg) => {
          const parsed = JSON.parse(msg.data) as StreamEvent;
          setEvents((prev) => [...prev, parsed]);
        };
        stream.addEventListener('done', () => {
          stream.close();
          resolve();
        });
        stream.onerror = () => {
          stream.close();
          resolve();
        };
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'unknown error');
    } finally {
      setIsRunning(false);
      setMessage('');
    }
  };

  return (
    <section className="space-y-4">
      <header>
        <h1 className="text-2xl font-semibold">Chat</h1>
        <p className="mt-2 text-sm text-muted">发送消息并实时查看任务事件流。</p>
      </header>

      <form onSubmit={onSubmit} className="space-y-3 rounded-lg border border-border bg-surface p-4">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="输入消息..."
          className="h-24 w-full rounded-md border border-border bg-surface2 p-3 text-sm text-text outline-none focus:border-primary"
        />
        <button
          type="submit"
          disabled={isRunning}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {isRunning ? 'Running...' : 'Send'}
        </button>
      </form>

      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="text-xs text-muted">conversation: {conversationId || '-'}</p>
        <p className="text-xs text-muted">task: {taskId || '-'}</p>
        {error ? <p className="mt-2 text-sm text-danger">{error}</p> : null}
        <div className="mt-3 space-y-2">
          {events.map((eventItem) => (
            <div key={eventItem.event_id} className="rounded border border-border bg-surface2 p-2 text-xs text-text">
              <div className="font-mono text-muted">{eventItem.type}</div>
              <pre className="mt-1 overflow-x-auto text-[11px]">{JSON.stringify(eventItem.payload, null, 2)}</pre>
            </div>
          ))}
        </div>
        {completionText ? (
          <div className="mt-3 rounded border border-primary/60 bg-primary/10 p-3 text-sm text-text">
            <strong>Completion:</strong> {completionText}
          </div>
        ) : null}
      </div>
    </section>
  );
}

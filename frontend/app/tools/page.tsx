'use client';

import { FormEvent, useEffect, useMemo, useState } from 'react';

type ToolSchema = {
  required?: string[];
  properties?: Record<string, unknown>;
};

type ToolItem = {
  name: string;
  version: string;
  risk: string;
  requires_approval: boolean;
  schema: ToolSchema;
  enabled: boolean;
};

type ExecutePayload = {
  tool_call_id?: string;
  status?: string;
  tool?: string;
  approval_id?: string;
  policy_reason?: string;
  result?: unknown;
  error?: string;
  idempotent_hit?: boolean;
  session_id?: string;
  session_workspace?: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';

export default function ToolsPage() {
  const [tools, setTools] = useState<ToolItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<ExecutePayload | null>(null);

  const [selectedTool, setSelectedTool] = useState('');
  const [argsText, setArgsText] = useState('{\n  "query": "hello"\n}');
  const [taskId, setTaskId] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [executionMode, setExecutionMode] = useState('');

  const selectedSpec = useMemo(() => tools.find((item) => item.name === selectedTool) ?? null, [tools, selectedTool]);

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

  const loadTools = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/v1/tools`);
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const payload = (await response.json()) as { tools: ToolItem[] };
      const list = payload.tools ?? [];
      setTools(list);
      if (!selectedTool && list.length > 0) {
        setSelectedTool(list[0].name);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadTools();
  }, []);

  const onExecute = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedTool) {
      return;
    }

    let parsedArgs: Record<string, unknown>;
    try {
      const decoded = JSON.parse(argsText);
      if (typeof decoded !== 'object' || decoded === null || Array.isArray(decoded)) {
        setError('args must be a JSON object');
        return;
      }
      parsedArgs = decoded as Record<string, unknown>;
    } catch {
      setError('args is not valid JSON');
      return;
    }

    setRunning(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/v1/tools:execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Actor-Role': 'owner',
          'X-Actor-Id': 'web-owner'
        },
        body: JSON.stringify({
          tool: selectedTool,
          args: parsedArgs,
          task_id: taskId.trim() || undefined,
          session_id: sessionId.trim() || undefined,
          execution_mode: executionMode || undefined
        })
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const payload = (await response.json()) as ExecutePayload;
      setResult(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'unknown error');
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Tools</h1>
          <p className="mt-2 text-sm text-muted">查看工具清单、schema，并可直接执行调试。</p>
        </div>
        <button onClick={() => void loadTools()} className="rounded-md border border-border px-3 py-2 text-sm text-text">
          {loading ? 'Loading...' : 'Refresh Tools'}
        </button>
      </header>

      {error ? <p className="text-sm text-danger">{error}</p> : null}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.2fr_1fr]">
        <div className="rounded-lg border border-border bg-surface p-4">
          <h2 className="text-sm font-medium">Tool Registry</h2>
          <div className="mt-3 max-h-[520px] space-y-2 overflow-auto">
            {tools.map((item) => (
              <button
                key={item.name}
                onClick={() => setSelectedTool(item.name)}
                className={`w-full rounded border p-3 text-left text-xs ${
                  selectedTool === item.name
                    ? 'border-primary bg-primary/10'
                    : 'border-border bg-surface2 hover:border-primary/60'
                }`}
              >
                <p className="font-mono text-text">{item.name}</p>
                <p className="mt-1 text-muted">
                  risk={item.risk} version={item.version} enabled={String(item.enabled)}
                </p>
                <p className="mt-1 text-muted">requires_approval={String(item.requires_approval)}</p>
              </button>
            ))}
            {tools.length === 0 ? <p className="text-xs text-muted">No tools loaded.</p> : null}
          </div>
        </div>

        <div className="space-y-4">
          <form onSubmit={onExecute} className="rounded-lg border border-border bg-surface p-4">
            <h2 className="text-sm font-medium">Execute Tool</h2>
            <div className="mt-3 space-y-2">
              <label className="block text-xs text-muted">
                tool
                <select
                  value={selectedTool}
                  onChange={(e) => setSelectedTool(e.target.value)}
                  className="mt-1 w-full rounded border border-border bg-surface2 px-2 py-2 text-sm text-text"
                >
                  <option value="">Select tool</option>
                  {tools.map((item) => (
                    <option key={item.name} value={item.name}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block text-xs text-muted">
                args (JSON object)
                <textarea
                  value={argsText}
                  onChange={(e) => setArgsText(e.target.value)}
                  className="mt-1 h-36 w-full rounded border border-border bg-surface2 p-2 text-xs text-text"
                />
              </label>

              <label className="block text-xs text-muted">
                task_id (optional)
                <input
                  value={taskId}
                  onChange={(e) => setTaskId(e.target.value)}
                  className="mt-1 w-full rounded border border-border bg-surface2 px-2 py-2 text-sm text-text"
                />
              </label>

              <label className="block text-xs text-muted">
                session_id (optional)
                <input
                  value={sessionId}
                  onChange={(e) => setSessionId(e.target.value)}
                  className="mt-1 w-full rounded border border-border bg-surface2 px-2 py-2 text-sm text-text"
                />
              </label>

              <label className="block text-xs text-muted">
                execution_mode (optional)
                <select
                  value={executionMode}
                  onChange={(e) => setExecutionMode(e.target.value)}
                  className="mt-1 w-full rounded border border-border bg-surface2 px-2 py-2 text-sm text-text"
                >
                  <option value="">default</option>
                  <option value="auto">auto</option>
                  <option value="local">local</option>
                  <option value="sandbox">sandbox</option>
                </select>
              </label>
            </div>
            <button type="submit" disabled={running || !selectedTool} className="mt-3 rounded bg-primary px-4 py-2 text-sm text-white disabled:opacity-60">
              {running ? 'Executing...' : 'Execute'}
            </button>
          </form>

          <div className="rounded-lg border border-border bg-surface p-4">
            <h2 className="text-sm font-medium">Selected Tool Schema</h2>
            {selectedSpec ? (
              <>
                <p className="mt-2 text-xs text-muted">required: {(selectedSpec.schema.required ?? []).join(', ') || '[none]'}</p>
                <pre className="mt-2 overflow-x-auto rounded border border-border bg-surface2 p-2 text-[11px] text-text">
                  {JSON.stringify(selectedSpec.schema, null, 2)}
                </pre>
              </>
            ) : (
              <p className="mt-2 text-xs text-muted">Select a tool to inspect schema.</p>
            )}
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-surface p-4">
        <h2 className="text-sm font-medium">Execution Result</h2>
        {result ? (
          <>
            {result.status === 'pending_approval' ? (
              <p className="mt-2 text-xs text-warning">
                pending approval: {result.approval_id} {result.policy_reason ? `(${result.policy_reason})` : ''}
              </p>
            ) : null}
            <pre className="mt-2 overflow-x-auto rounded border border-border bg-surface2 p-2 text-[11px] text-text">
              {JSON.stringify(result, null, 2)}
            </pre>
          </>
        ) : (
          <p className="mt-2 text-xs text-muted">No execution output yet.</p>
        )}
      </div>
    </section>
  );
}


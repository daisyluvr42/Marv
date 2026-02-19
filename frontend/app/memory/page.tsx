'use client';

import { FormEvent, useState } from 'react';

type MemoryResult = {
  id: string;
  content: string;
  score: number;
};

type Candidate = {
  id: string;
  content: string;
  status: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';

export default function MemoryPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<MemoryResult[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);

  const loadCandidates = async () => {
    const response = await fetch(`${API_BASE}/v1/memory/candidates?status=pending`);
    const payload = (await response.json()) as { candidates: Candidate[] };
    setCandidates(payload.candidates ?? []);
  };

  const onQuery = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const response = await fetch(`${API_BASE}/v1/memory/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scope_type: 'user',
        scope_id: 'u1',
        query,
        top_k: 5
      })
    });
    const payload = (await response.json()) as { results: MemoryResult[] };
    setResults(payload.results ?? []);
  };

  const decide = async (candidateId: string, action: 'approve' | 'reject') => {
    await fetch(`${API_BASE}/v1/memory/candidates/${candidateId}:${action}`, {
      method: 'POST',
      headers: {
        'X-Actor-Role': 'owner',
        'X-Actor-Id': 'web-owner'
      }
    });
    await loadCandidates();
  };

  return (
    <section className="space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Memory</h1>
          <p className="mt-2 text-sm text-muted">记忆查询与候选审批。</p>
        </div>
        <button onClick={() => void loadCandidates()} className="rounded-md border border-border px-3 py-2 text-sm">
          Load Candidates
        </button>
      </header>

      <form onSubmit={onQuery} className="flex gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="查询记忆"
          className="flex-1 rounded border border-border bg-surface px-3 py-2 text-sm"
        />
        <button className="rounded bg-primary px-3 py-2 text-sm text-white">Query</button>
      </form>

      <div className="rounded border border-border bg-surface p-3">
        <h2 className="text-sm font-medium">Results</h2>
        <div className="mt-2 space-y-2">
          {results.map((item) => (
            <div key={item.id} className="rounded border border-border bg-surface2 p-2 text-xs">
              <p>{item.content}</p>
              <p className="text-muted">score: {item.score.toFixed(3)}</p>
            </div>
          ))}
          {results.length === 0 ? <p className="text-xs text-muted">No query results.</p> : null}
        </div>
      </div>

      <div className="rounded border border-border bg-surface p-3">
        <h2 className="text-sm font-medium">Candidates</h2>
        <div className="mt-2 space-y-2">
          {candidates.map((item) => (
            <div key={item.id} className="rounded border border-border bg-surface2 p-2 text-xs">
              <p className="font-mono text-muted">{item.id}</p>
              <p className="mt-1">{item.content}</p>
              <div className="mt-2 flex gap-2">
                <button onClick={() => void decide(item.id, 'approve')} className="rounded bg-success/20 px-2 py-1 text-success">
                  Approve
                </button>
                <button onClick={() => void decide(item.id, 'reject')} className="rounded bg-danger/20 px-2 py-1 text-danger">
                  Reject
                </button>
              </div>
            </div>
          ))}
          {candidates.length === 0 ? <p className="text-xs text-muted">No pending candidates.</p> : null}
        </div>
      </div>
    </section>
  );
}

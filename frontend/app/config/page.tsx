'use client';

import { FormEvent, useState } from 'react';

type Revision = {
  revision: string;
  risk_level: string;
  status: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';

export default function ConfigPage() {
  const [text, setText] = useState('更简洁');
  const [proposalId, setProposalId] = useState('');
  const [revisions, setRevisions] = useState<Revision[]>([]);
  const [effectiveStyle, setEffectiveStyle] = useState('');

  const loadRevisions = async () => {
    const response = await fetch(`${API_BASE}/v1/config/revisions?scope_type=channel&scope_id=web:default`);
    const payload = (await response.json()) as {
      revisions: Revision[];
      effective_config?: { response_style?: string };
    };
    setRevisions(payload.revisions ?? []);
    setEffectiveStyle(payload.effective_config?.response_style ?? '');
  };

  const propose = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const response = await fetch(`${API_BASE}/v1/config/patches:propose`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Actor-Role': 'owner',
        'X-Actor-Id': 'web-owner'
      },
      body: JSON.stringify({
        natural_language: text,
        scope_type: 'channel',
        scope_id: 'web:default'
      })
    });
    const payload = (await response.json()) as { proposal_id: string };
    setProposalId(payload.proposal_id);
  };

  const commit = async () => {
    if (!proposalId) return;
    await fetch(`${API_BASE}/v1/config/patches:commit`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Actor-Role': 'owner',
        'X-Actor-Id': 'web-owner'
      },
      body: JSON.stringify({ proposal_id: proposalId })
    });
    await loadRevisions();
  };

  const rollback = async (revision: string) => {
    await fetch(`${API_BASE}/v1/config/revisions:rollback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Actor-Role': 'owner',
        'X-Actor-Id': 'web-owner'
      },
      body: JSON.stringify({ revision })
    });
    await loadRevisions();
  };

  return (
    <section className="space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Config</h1>
          <p className="mt-2 text-sm text-muted">补丁提案、提交与回滚。</p>
        </div>
        <button onClick={() => void loadRevisions()} className="rounded-md border border-border px-3 py-2 text-sm">
          Load Revisions
        </button>
      </header>

      <form onSubmit={propose} className="flex gap-2">
        <input value={text} onChange={(e) => setText(e.target.value)} className="flex-1 rounded border border-border bg-surface px-3 py-2 text-sm" />
        <button className="rounded bg-primary px-3 py-2 text-sm text-white">Propose</button>
      </form>

      <div className="flex gap-2">
        <button onClick={() => void commit()} className="rounded bg-success/20 px-3 py-2 text-sm text-success">
          Commit Proposal
        </button>
        <p className="self-center text-xs text-muted">proposal: {proposalId || '-'}</p>
      </div>

      <div className="rounded border border-border bg-surface p-3">
        <p className="text-xs text-muted">effective response_style: {effectiveStyle || '-'}</p>
        <div className="mt-2 space-y-2">
          {revisions.map((item) => (
            <div key={item.revision} className="rounded border border-border bg-surface2 p-2 text-xs">
              <p className="font-mono text-muted">{item.revision}</p>
              <p className="mt-1">risk={item.risk_level} status={item.status}</p>
              {item.status === 'committed' ? (
                <button onClick={() => void rollback(item.revision)} className="mt-2 rounded bg-warning/20 px-2 py-1 text-warning">
                  Rollback
                </button>
              ) : null}
            </div>
          ))}
          {revisions.length === 0 ? <p className="text-xs text-muted">No revisions.</p> : null}
        </div>
      </div>
    </section>
  );
}

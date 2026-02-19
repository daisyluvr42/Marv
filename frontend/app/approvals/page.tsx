'use client';

import { useEffect, useState } from 'react';

type ApprovalItem = {
  approval_id: string;
  type: string;
  status: string;
  summary: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';

export default function ApprovalsPage() {
  const [items, setItems] = useState<ApprovalItem[]>([]);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    const response = await fetch(`${API_BASE}/v1/approvals?status=pending`);
    const payload = (await response.json()) as { approvals: ApprovalItem[] };
    setItems(payload.approvals ?? []);
    setLoading(false);
  };

  const decide = async (approvalId: string, action: 'approve' | 'reject') => {
    await fetch(`${API_BASE}/v1/approvals/${approvalId}:${action}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Actor-Role': 'owner',
        'X-Actor-Id': 'web-owner'
      },
      body: '{}'
    });
    await load();
  };

  useEffect(() => {
    void load();
  }, []);

  return (
    <section className="space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Approvals</h1>
          <p className="mt-2 text-sm text-muted">处理待审批的高风险操作。</p>
        </div>
        <button onClick={() => void load()} className="rounded-md border border-border px-3 py-2 text-sm text-text">
          Refresh
        </button>
      </header>
      <div className="space-y-2">
        {loading ? <p className="text-sm text-muted">Loading...</p> : null}
        {items.map((item) => (
          <div key={item.approval_id} className="rounded border border-border bg-surface p-3">
            <p className="font-mono text-xs text-muted">{item.approval_id}</p>
            <p className="mt-1 text-sm text-text">{item.summary}</p>
            <div className="mt-2 flex gap-2">
              <button
                onClick={() => void decide(item.approval_id, 'approve')}
                className="rounded bg-success/20 px-3 py-1 text-xs text-success"
              >
                Approve
              </button>
              <button
                onClick={() => void decide(item.approval_id, 'reject')}
                className="rounded bg-danger/20 px-3 py-1 text-xs text-danger"
              >
                Reject
              </button>
            </div>
          </div>
        ))}
        {!loading && items.length === 0 ? <p className="text-sm text-muted">No pending approvals.</p> : null}
      </div>
    </section>
  );
}

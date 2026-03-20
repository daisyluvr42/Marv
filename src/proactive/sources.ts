import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../core/config/paths.js";
import { withFileLock } from "../infra/file-lock.js";
import { createAsyncLock } from "../infra/json-files.js";

// ── Types ──────────────────────────────────────────────────────────────

export type InfoSourceKind = "rss" | "web" | "email" | "api";

export type InfoSource = {
  id: string;
  kind: InfoSourceKind;
  label: string;
  url?: string;
  /** Source-specific configuration (e.g. query params, auth headers). */
  config?: Record<string, unknown>;
  pollIntervalMs: number;
  lastPolledAt: number;
  enabled: boolean;
  createdAt: number;
  updatedAt: number;
};

export type SourceEvent = {
  id: string;
  sourceId: string;
  summary: string;
  detail?: string;
  /** ISO timestamp of the external event. */
  eventTime?: string;
  createdAt: number;
};

export type SourceStoreData = {
  sources: InfoSource[];
  /** Normalized events from the most recent poll, used for deduplication. */
  recentEvents: SourceEvent[];
};

// ── File lock infrastructure ───────────────────────────────────────────

const SOURCE_STORE_LOCK_OPTIONS = {
  retries: { retries: 8, factor: 2, minTimeout: 25, maxTimeout: 1_000, randomize: true },
  stale: 10_000,
} as const;

const withSourceStoreProcessLock = createAsyncLock();

const DEFAULT_POLL_INTERVAL_MS = 30 * 60_000; // 30 min
const MAX_RECENT_EVENTS = 200;

function resolveSourceStorePath(agentId: string): string {
  return path.join(resolveStateDir(process.env), "proactive", agentId, "sources.json");
}

async function readSourceStoreFromPath(filePath: string): Promise<SourceStoreData> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw) as SourceStoreData;
    return {
      sources: Array.isArray(parsed.sources) ? [...parsed.sources] : [],
      recentEvents: Array.isArray(parsed.recentEvents) ? [...parsed.recentEvents] : [],
    };
  } catch {
    return { sources: [], recentEvents: [] };
  }
}

async function writeSourceStoreToPath(filePath: string, data: SourceStoreData): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, "utf-8");
}

async function withSourceStoreLock<T>(
  agentId: string,
  fn: (filePath: string) => Promise<T>,
): Promise<T> {
  const filePath = resolveSourceStorePath(agentId);
  return await withSourceStoreProcessLock(
    async () =>
      await withFileLock(filePath, SOURCE_STORE_LOCK_OPTIONS, async () => await fn(filePath)),
  );
}

// ── Public API ─────────────────────────────────────────────────────────

export async function readSourceStore(agentId: string): Promise<SourceStoreData> {
  const filePath = resolveSourceStorePath(agentId);
  return await readSourceStoreFromPath(filePath);
}

export async function addSource(
  agentId: string,
  params: {
    kind: InfoSourceKind;
    label: string;
    url?: string;
    config?: Record<string, unknown>;
    pollIntervalMs?: number;
  },
): Promise<InfoSource> {
  return await withSourceStoreLock(agentId, async (filePath) => {
    const data = await readSourceStoreFromPath(filePath);
    const now = Date.now();
    const source: InfoSource = {
      id: `src_${crypto.randomUUID().replace(/-/g, "")}`,
      kind: params.kind,
      label: params.label,
      url: params.url,
      config: params.config,
      pollIntervalMs: params.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS,
      lastPolledAt: 0,
      enabled: true,
      createdAt: now,
      updatedAt: now,
    };
    data.sources.push(source);
    await writeSourceStoreToPath(filePath, data);
    return source;
  });
}

export async function removeSource(agentId: string, sourceId: string): Promise<boolean> {
  return await withSourceStoreLock(agentId, async (filePath) => {
    const data = await readSourceStoreFromPath(filePath);
    const before = data.sources.length;
    data.sources = data.sources.filter((s) => s.id !== sourceId);
    if (data.sources.length === before) {
      return false;
    }
    await writeSourceStoreToPath(filePath, data);
    return true;
  });
}

export async function updateSource(
  agentId: string,
  sourceId: string,
  patch: Partial<Pick<InfoSource, "label" | "url" | "config" | "pollIntervalMs" | "enabled">>,
): Promise<InfoSource | null> {
  return await withSourceStoreLock(agentId, async (filePath) => {
    const data = await readSourceStoreFromPath(filePath);
    const source = data.sources.find((s) => s.id === sourceId);
    if (!source) {
      return null;
    }
    if (patch.label !== undefined) {
      source.label = patch.label;
    }
    if (patch.url !== undefined) {
      source.url = patch.url;
    }
    if (patch.config !== undefined) {
      source.config = patch.config;
    }
    if (patch.pollIntervalMs !== undefined) {
      source.pollIntervalMs = patch.pollIntervalMs;
    }
    if (patch.enabled !== undefined) {
      source.enabled = patch.enabled;
    }
    source.updatedAt = Date.now();
    await writeSourceStoreToPath(filePath, data);
    return { ...source };
  });
}

export async function listSources(
  agentId: string,
  filter?: { enabled?: boolean },
): Promise<InfoSource[]> {
  const data = await readSourceStore(agentId);
  let sources = data.sources;
  if (filter?.enabled !== undefined) {
    sources = sources.filter((s) => s.enabled === filter.enabled);
  }
  return sources;
}

/** Mark a source as polled and record new events (deduplicated by event ID). */
export async function recordPollResult(
  agentId: string,
  sourceId: string,
  events: Omit<SourceEvent, "createdAt">[],
): Promise<{ newEvents: SourceEvent[] }> {
  return await withSourceStoreLock(agentId, async (filePath) => {
    const data = await readSourceStoreFromPath(filePath);
    const source = data.sources.find((s) => s.id === sourceId);
    if (source) {
      source.lastPolledAt = Date.now();
      source.updatedAt = Date.now();
    }

    const existingIds = new Set(data.recentEvents.map((e) => e.id));
    const newEvents: SourceEvent[] = [];
    for (const evt of events) {
      if (existingIds.has(evt.id)) {
        continue;
      }
      const full: SourceEvent = { ...evt, createdAt: Date.now() };
      newEvents.push(full);
      data.recentEvents.push(full);
    }

    // Trim old events to prevent unbounded growth.
    if (data.recentEvents.length > MAX_RECENT_EVENTS) {
      data.recentEvents = data.recentEvents.slice(-MAX_RECENT_EVENTS);
    }

    await writeSourceStoreToPath(filePath, data);
    return { newEvents };
  });
}

/** Get sources that are due for polling (past their poll interval). */
export async function getSourcesDueForPolling(agentId: string): Promise<InfoSource[]> {
  const data = await readSourceStore(agentId);
  const now = Date.now();
  return data.sources.filter((s) => s.enabled && now - s.lastPolledAt >= s.pollIntervalMs);
}

/** Get recent events, optionally filtered by source. */
export async function getRecentEvents(
  agentId: string,
  filter?: { sourceId?: string },
): Promise<SourceEvent[]> {
  const data = await readSourceStore(agentId);
  let events = data.recentEvents;
  if (filter?.sourceId) {
    events = events.filter((e) => e.sourceId === filter.sourceId);
  }
  return events;
}

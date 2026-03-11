import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../core/config/paths.js";
import { withFileLock } from "../infra/file-lock.js";
import { createAsyncLock } from "../infra/json-files.js";

export type DigestBufferEntry = {
  id: string;
  timestamp: number;
  source: string;
  summary: string;
  detail?: string;
  urgency: "normal" | "urgent";
  delivered: boolean;
  jobId?: string;
};

export type DigestBuffer = {
  entries: DigestBufferEntry[];
  lastFlushAt: number;
};

const DIGEST_BUFFER_LOCK_OPTIONS = {
  retries: {
    retries: 8,
    factor: 2,
    minTimeout: 25,
    maxTimeout: 1_000,
    randomize: true,
  },
  stale: 10_000,
} as const;
const withDigestBufferProcessLock = createAsyncLock();

export async function readDigestBuffer(agentId: string): Promise<DigestBuffer> {
  const filePath = resolveDigestBufferPath(agentId);
  return await readDigestBufferFromPath(filePath);
}

async function readDigestBufferFromPath(filePath: string): Promise<DigestBuffer> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw) as DigestBuffer;
    return {
      entries: Array.isArray(parsed.entries) ? [...parsed.entries] : [],
      lastFlushAt: typeof parsed.lastFlushAt === "number" ? parsed.lastFlushAt : 0,
    };
  } catch {
    return createEmptyDigestBuffer();
  }
}

export async function appendToDigestBuffer(
  agentId: string,
  entry: Omit<DigestBufferEntry, "id" | "timestamp" | "delivered">,
): Promise<DigestBufferEntry> {
  return await withDigestBufferLock(agentId, async (filePath) => {
    const buffer = await readDigestBufferFromPath(filePath);
    const nextEntry: DigestBufferEntry = {
      id: `digest_${crypto.randomUUID().replace(/-/g, "")}`,
      timestamp: Date.now(),
      delivered: false,
      ...entry,
    };
    buffer.entries.push(nextEntry);
    await writeDigestBufferToPath(filePath, buffer);
    return nextEntry;
  });
}

export async function flushDigestBuffer(agentId: string): Promise<DigestBufferEntry[]> {
  return await withDigestBufferLock(agentId, async (filePath) => {
    const buffer = await readDigestBufferFromPath(filePath);
    const pending = buffer.entries.filter((entry) => !entry.delivered);
    if (pending.length === 0) {
      return [];
    }
    const deliveredIds = new Set(pending.map((entry) => entry.id));
    await writeDigestBufferToPath(filePath, {
      entries: buffer.entries.map((entry) =>
        deliveredIds.has(entry.id) ? { ...entry, delivered: true } : entry,
      ),
      lastFlushAt: Date.now(),
    });
    return pending;
  });
}

export async function clearDeliveredEntries(agentId: string): Promise<void> {
  await withDigestBufferLock(agentId, async (filePath) => {
    const buffer = await readDigestBufferFromPath(filePath);
    await writeDigestBufferToPath(filePath, {
      entries: buffer.entries.filter((entry) => !entry.delivered),
      lastFlushAt: buffer.lastFlushAt,
    });
  });
}

function resolveDigestBufferPath(agentId: string): string {
  return path.join(resolveStateDir(process.env), "proactive", agentId, "buffer.json");
}

function createEmptyDigestBuffer(): DigestBuffer {
  return {
    entries: [],
    lastFlushAt: 0,
  };
}

async function writeDigestBufferToPath(filePath: string, buffer: DigestBuffer): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(buffer, null, 2)}\n`, "utf-8");
}

async function withDigestBufferLock<T>(
  agentId: string,
  fn: (filePath: string) => Promise<T>,
): Promise<T> {
  const filePath = resolveDigestBufferPath(agentId);
  return await withDigestBufferProcessLock(
    async () =>
      await withFileLock(filePath, DIGEST_BUFFER_LOCK_OPTIONS, async () => await fn(filePath)),
  );
}

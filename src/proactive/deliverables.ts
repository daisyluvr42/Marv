import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../core/config/paths.js";
import { withFileLock } from "../infra/file-lock.js";
import { createAsyncLock } from "../infra/json-files.js";

// ── Types ──────────────────────────────────────────────────────────────

export type DeliverableKind = "report" | "article" | "tool" | "plan" | "analysis" | "other";

/**
 * Lifecycle:  stored → announced  (or → failed on notification error)
 * Completing a task should register a deliverable as "stored".
 * A separate delivery step transitions it to "announced".
 * This two-step model prevents duplicate notifications on retry.
 */
export type DeliverableStatus = "stored" | "announced" | "failed";

export type Deliverable = {
  id: string;
  taskId?: string;
  goalId?: string;
  title: string;
  kind: DeliverableKind;
  /** Relative path within the deliverables directory. */
  filePath?: string;
  /** SHA-256 hash of the content for deduplication. */
  contentHash?: string;
  status: DeliverableStatus;
  createdAt: number;
  announcedAt?: number;
  /** Error from the last announcement attempt. */
  lastAnnouncementError?: string;
};

export type DeliverableStoreData = {
  deliverables: Deliverable[];
};

// ── File lock infrastructure ───────────────────────────────────────────

const DELIVERABLE_LOCK_OPTIONS = {
  retries: { retries: 8, factor: 2, minTimeout: 25, maxTimeout: 1_000, randomize: true },
  stale: 10_000,
} as const;

const withDeliverableProcessLock = createAsyncLock();

function resolveDeliverableIndexPath(agentId: string): string {
  return path.join(
    resolveStateDir(process.env),
    "proactive",
    agentId,
    "deliverables",
    "index.json",
  );
}

export function resolveDeliverableDir(agentId: string): string {
  return path.join(resolveStateDir(process.env), "proactive", agentId, "deliverables");
}

async function readDeliverableStoreFromPath(filePath: string): Promise<DeliverableStoreData> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw) as DeliverableStoreData;
    return {
      deliverables: Array.isArray(parsed.deliverables) ? [...parsed.deliverables] : [],
    };
  } catch {
    return { deliverables: [] };
  }
}

async function writeDeliverableStoreToPath(
  filePath: string,
  data: DeliverableStoreData,
): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, "utf-8");
}

async function withDeliverableLock<T>(
  agentId: string,
  fn: (filePath: string) => Promise<T>,
): Promise<T> {
  const filePath = resolveDeliverableIndexPath(agentId);
  return await withDeliverableProcessLock(
    async () =>
      await withFileLock(filePath, DELIVERABLE_LOCK_OPTIONS, async () => await fn(filePath)),
  );
}

// ── Public API ─────────────────────────────────────────────────────────

export async function readDeliverableStore(agentId: string): Promise<DeliverableStoreData> {
  const filePath = resolveDeliverableIndexPath(agentId);
  return await readDeliverableStoreFromPath(filePath);
}

/**
 * Register a new deliverable as "stored". Does not send any notification.
 * Deduplicates by contentHash if provided — returns existing deliverable if hash matches.
 */
export async function registerDeliverable(
  agentId: string,
  params: {
    taskId?: string;
    goalId?: string;
    title: string;
    kind: DeliverableKind;
    filePath?: string;
    contentHash?: string;
  },
): Promise<{ deliverable: Deliverable; created: boolean }> {
  return await withDeliverableLock(agentId, async (indexPath) => {
    const data = await readDeliverableStoreFromPath(indexPath);

    // Deduplicate by contentHash if provided.
    if (params.contentHash) {
      const existing = data.deliverables.find(
        (d) => d.contentHash === params.contentHash && d.status !== "failed",
      );
      if (existing) {
        return { deliverable: { ...existing }, created: false };
      }
    }

    // Deduplicate by taskId — prevent auto-registration from creating a second
    // deliverable when the LLM already registered one during task execution.
    if (params.taskId) {
      const existing = data.deliverables.find(
        (d) => d.taskId === params.taskId && d.status !== "failed",
      );
      if (existing) {
        return { deliverable: { ...existing }, created: false };
      }
    }

    const now = Date.now();
    const deliverable: Deliverable = {
      id: `deliv_${crypto.randomUUID().replace(/-/g, "")}`,
      taskId: params.taskId,
      goalId: params.goalId,
      title: params.title,
      kind: params.kind,
      filePath: params.filePath,
      contentHash: params.contentHash,
      status: "stored",
      createdAt: now,
    };
    data.deliverables.push(deliverable);
    await writeDeliverableStoreToPath(indexPath, data);
    return { deliverable, created: true };
  });
}

/** Mark a deliverable as announced (notification sent successfully). */
export async function markAnnounced(agentId: string, deliverableId: string): Promise<void> {
  await withDeliverableLock(agentId, async (indexPath) => {
    const data = await readDeliverableStoreFromPath(indexPath);
    const d = data.deliverables.find((x) => x.id === deliverableId);
    if (!d) {
      return;
    }
    d.status = "announced";
    d.announcedAt = Date.now();
    d.lastAnnouncementError = undefined;
    await writeDeliverableStoreToPath(indexPath, data);
  });
}

/** Mark a deliverable announcement as failed with an error. */
export async function markAnnouncementFailed(
  agentId: string,
  deliverableId: string,
  error: string,
): Promise<void> {
  await withDeliverableLock(agentId, async (indexPath) => {
    const data = await readDeliverableStoreFromPath(indexPath);
    const d = data.deliverables.find((x) => x.id === deliverableId);
    if (!d) {
      return;
    }
    d.status = "failed";
    d.lastAnnouncementError = error;
    await writeDeliverableStoreToPath(indexPath, data);
  });
}

/** List deliverables, optionally filtered by status. */
export async function listDeliverables(
  agentId: string,
  filter?: { status?: DeliverableStatus },
): Promise<Deliverable[]> {
  const data = await readDeliverableStore(agentId);
  let deliverables = data.deliverables;
  if (filter?.status) {
    deliverables = deliverables.filter((d) => d.status === filter.status);
  }
  return deliverables;
}

/** Get unannounced deliverables that need notification. */
export async function getPendingAnnouncements(agentId: string): Promise<Deliverable[]> {
  const data = await readDeliverableStore(agentId);
  return data.deliverables.filter((d) => d.status === "stored");
}

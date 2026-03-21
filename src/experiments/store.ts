import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../core/config/paths.js";
import { withFileLock } from "../infra/file-lock.js";
import { createAsyncLock } from "../infra/json-files.js";
import type { ExperimentState, ExperimentStatus } from "./types.js";

// ── Types ──────────────────────────────────────────────────────────────

export type ExperimentStoreData = {
  experiments: ExperimentState[];
};

// ── File lock infrastructure (mirrors proactive/task-queue.ts) ─────────

const EXPERIMENT_STORE_LOCK_OPTIONS = {
  retries: {
    retries: 8,
    factor: 2,
    minTimeout: 25,
    maxTimeout: 1_000,
    randomize: true,
  },
  stale: 10_000,
} as const;

const withExperimentStoreProcessLock = createAsyncLock();

function resolveExperimentStorePath(agentId: string): string {
  return path.join(resolveStateDir(process.env), "proactive", agentId, "experiments.json");
}

async function readStoreFromPath(filePath: string): Promise<ExperimentStoreData> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw) as ExperimentStoreData;
    return {
      experiments: Array.isArray(parsed.experiments) ? [...parsed.experiments] : [],
    };
  } catch {
    return { experiments: [] };
  }
}

async function writeStoreToPath(filePath: string, data: ExperimentStoreData): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, "utf-8");
}

async function withStoreLock<T>(agentId: string, fn: (filePath: string) => Promise<T>): Promise<T> {
  const filePath = resolveExperimentStorePath(agentId);
  return await withExperimentStoreProcessLock(
    async () =>
      await withFileLock(filePath, EXPERIMENT_STORE_LOCK_OPTIONS, async () => await fn(filePath)),
  );
}

// ── Public API ─────────────────────────────────────────────────────────

export async function readExperimentStore(agentId: string): Promise<ExperimentStoreData> {
  const filePath = resolveExperimentStorePath(agentId);
  return await readStoreFromPath(filePath);
}

/** Create a new experiment and add it to the store. Returns the experiment ID. */
export async function createExperiment(agentId: string, state: ExperimentState): Promise<string> {
  return await withStoreLock(agentId, async (filePath) => {
    const data = await readStoreFromPath(filePath);
    data.experiments.push(state);
    await writeStoreToPath(filePath, data);
    return state.spec.id;
  });
}

/** Update an existing experiment in the store. */
export async function updateExperiment(
  agentId: string,
  experimentId: string,
  updater: (state: ExperimentState) => ExperimentState,
): Promise<void> {
  await withStoreLock(agentId, async (filePath) => {
    const data = await readStoreFromPath(filePath);
    const idx = data.experiments.findIndex((e) => e.spec.id === experimentId);
    if (idx < 0) {
      return;
    }
    data.experiments[idx] = updater(data.experiments[idx]);
    await writeStoreToPath(filePath, data);
  });
}

/** Get a single experiment by ID. */
export async function getExperiment(
  agentId: string,
  experimentId: string,
): Promise<ExperimentState | undefined> {
  const data = await readExperimentStore(agentId);
  return data.experiments.find((e) => e.spec.id === experimentId);
}

/** List experiments, optionally filtered by status. */
export async function listExperiments(
  agentId: string,
  statusFilter?: ExperimentStatus,
): Promise<ExperimentState[]> {
  const data = await readExperimentStore(agentId);
  if (!statusFilter) {
    return data.experiments;
  }
  return data.experiments.filter((e) => e.status === statusFilter);
}

/** Remove experiments older than maxAgeMs. */
export async function pruneOldExperiments(agentId: string, maxAgeMs: number): Promise<number> {
  return await withStoreLock(agentId, async (filePath) => {
    const data = await readStoreFromPath(filePath);
    const cutoff = Date.now() - maxAgeMs;
    const before = data.experiments.length;
    data.experiments = data.experiments.filter((e) => !e.completedAt || e.completedAt > cutoff);
    const removed = before - data.experiments.length;
    if (removed > 0) {
      await writeStoreToPath(filePath, data);
    }
    return removed;
  });
}

/** Generate a unique experiment ID. */
export function generateExperimentId(): string {
  return `exp_${crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`;
}

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../core/config/paths.js";
import { withFileLock } from "../infra/file-lock.js";
import { createAsyncLock } from "../infra/json-files.js";

// ── Types ──────────────────────────────────────────────────────────────

export type GoalPriority = "high" | "normal" | "low";

export type GoalStatus = "active" | "paused" | "completed";

export type ProactiveGoal = {
  id: string;
  title: string;
  description: string;
  priority: GoalPriority;
  status: GoalStatus;
  createdAt: number;
  updatedAt: number;
};

export type GoalStoreData = {
  goals: ProactiveGoal[];
};

// ── File lock infrastructure ───────────────────────────────────────────

const GOAL_STORE_LOCK_OPTIONS = {
  retries: {
    retries: 8,
    factor: 2,
    minTimeout: 25,
    maxTimeout: 1_000,
    randomize: true,
  },
  stale: 10_000,
} as const;

const withGoalStoreProcessLock = createAsyncLock();

function resolveGoalStorePath(agentId: string): string {
  return path.join(resolveStateDir(process.env), "proactive", agentId, "goals.json");
}

async function readGoalStoreFromPath(filePath: string): Promise<GoalStoreData> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw) as GoalStoreData;
    return {
      goals: Array.isArray(parsed.goals) ? [...parsed.goals] : [],
    };
  } catch {
    return { goals: [] };
  }
}

async function writeGoalStoreToPath(filePath: string, data: GoalStoreData): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, "utf-8");
}

async function withGoalStoreLock<T>(
  agentId: string,
  fn: (filePath: string) => Promise<T>,
): Promise<T> {
  const filePath = resolveGoalStorePath(agentId);
  return await withGoalStoreProcessLock(
    async () =>
      await withFileLock(filePath, GOAL_STORE_LOCK_OPTIONS, async () => await fn(filePath)),
  );
}

// ── Public API ─────────────────────────────────────────────────────────

export async function readGoalStore(agentId: string): Promise<GoalStoreData> {
  const filePath = resolveGoalStorePath(agentId);
  return await readGoalStoreFromPath(filePath);
}

export async function addGoal(
  agentId: string,
  params: { title: string; description: string; priority?: GoalPriority },
): Promise<ProactiveGoal> {
  return await withGoalStoreLock(agentId, async (filePath) => {
    const data = await readGoalStoreFromPath(filePath);
    const now = Date.now();
    const goal: ProactiveGoal = {
      id: `goal_${crypto.randomUUID().replace(/-/g, "")}`,
      title: params.title,
      description: params.description,
      priority: params.priority ?? "normal",
      status: "active",
      createdAt: now,
      updatedAt: now,
    };
    data.goals.push(goal);
    await writeGoalStoreToPath(filePath, data);
    return goal;
  });
}

export async function updateGoal(
  agentId: string,
  goalId: string,
  patch: Partial<Pick<ProactiveGoal, "title" | "description" | "priority" | "status">>,
): Promise<ProactiveGoal | null> {
  return await withGoalStoreLock(agentId, async (filePath) => {
    const data = await readGoalStoreFromPath(filePath);
    const goal = data.goals.find((g) => g.id === goalId);
    if (!goal) {
      return null;
    }
    if (patch.title !== undefined) {
      goal.title = patch.title;
    }
    if (patch.description !== undefined) {
      goal.description = patch.description;
    }
    if (patch.priority !== undefined) {
      goal.priority = patch.priority;
    }
    if (patch.status !== undefined) {
      goal.status = patch.status;
    }
    goal.updatedAt = Date.now();
    await writeGoalStoreToPath(filePath, data);
    return { ...goal };
  });
}

export async function listGoals(
  agentId: string,
  filter?: { status?: GoalStatus },
): Promise<ProactiveGoal[]> {
  const data = await readGoalStore(agentId);
  let goals = data.goals;
  if (filter?.status) {
    goals = goals.filter((g) => g.status === filter.status);
  }
  return goals;
}

export async function removeGoal(agentId: string, goalId: string): Promise<boolean> {
  return await withGoalStoreLock(agentId, async (filePath) => {
    const data = await readGoalStoreFromPath(filePath);
    const before = data.goals.length;
    data.goals = data.goals.filter((g) => g.id !== goalId);
    if (data.goals.length === before) {
      return false;
    }
    await writeGoalStoreToPath(filePath, data);
    return true;
  });
}

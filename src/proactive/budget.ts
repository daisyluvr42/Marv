import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../core/config/paths.js";
import { withFileLock } from "../infra/file-lock.js";
import { createAsyncLock } from "../infra/json-files.js";

// ── Types ──────────────────────────────────────────────────────────────

export type DailyTokenRecord = {
  /** Date string in YYYY-MM-DD format. */
  date: string;
  /** Total tokens consumed on this date. */
  tokens: number;
};

export type BudgetStoreData = {
  /** Rolling window of daily token records (kept for last 30 days). */
  daily: DailyTokenRecord[];
};

export type BudgetStatus = {
  todayTokens: number;
  dailyLimit: number;
  exhausted: boolean;
  /** Remaining tokens before hitting the limit. 0 when unlimited (limit=0). */
  remaining: number;
};

// ── File lock infrastructure ───────────────────────────────────────────

const BUDGET_LOCK_OPTIONS = {
  retries: { retries: 8, factor: 2, minTimeout: 25, maxTimeout: 1_000, randomize: true },
  stale: 10_000,
} as const;

const withBudgetProcessLock = createAsyncLock();

const MAX_DAILY_RECORDS = 30;

function resolveBudgetPath(agentId: string): string {
  return path.join(resolveStateDir(process.env), "proactive", agentId, "budget.json");
}

function todayKey(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

async function readBudgetFromPath(filePath: string): Promise<BudgetStoreData> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw) as BudgetStoreData;
    return {
      daily: Array.isArray(parsed.daily) ? [...parsed.daily] : [],
    };
  } catch {
    return { daily: [] };
  }
}

async function writeBudgetToPath(filePath: string, data: BudgetStoreData): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, "utf-8");
}

async function withBudgetLock<T>(
  agentId: string,
  fn: (filePath: string) => Promise<T>,
): Promise<T> {
  const filePath = resolveBudgetPath(agentId);
  return await withBudgetProcessLock(
    async () => await withFileLock(filePath, BUDGET_LOCK_OPTIONS, async () => await fn(filePath)),
  );
}

// ── Public API ─────────────────────────────────────────────────────────

/** Record token usage for today. */
export async function recordTokenUsage(agentId: string, tokens: number): Promise<void> {
  if (tokens <= 0) {
    return;
  }
  await withBudgetLock(agentId, async (filePath) => {
    const data = await readBudgetFromPath(filePath);
    const today = todayKey();
    const existing = data.daily.find((d) => d.date === today);
    if (existing) {
      existing.tokens += tokens;
    } else {
      data.daily.push({ date: today, tokens });
    }
    // Prune old records.
    if (data.daily.length > MAX_DAILY_RECORDS) {
      data.daily = data.daily.slice(-MAX_DAILY_RECORDS);
    }
    await writeBudgetToPath(filePath, data);
  });
}

/** Get today's token usage. */
export async function getTodayTokenUsage(agentId: string): Promise<number> {
  const filePath = resolveBudgetPath(agentId);
  const data = await readBudgetFromPath(filePath);
  const today = todayKey();
  return data.daily.find((d) => d.date === today)?.tokens ?? 0;
}

/** Check budget status against a daily limit. 0 = unlimited. */
export async function getBudgetStatus(agentId: string, dailyLimit: number): Promise<BudgetStatus> {
  const todayTokens = await getTodayTokenUsage(agentId);
  const exhausted = dailyLimit > 0 && todayTokens >= dailyLimit;
  const remaining = dailyLimit > 0 ? Math.max(0, dailyLimit - todayTokens) : 0;
  return { todayTokens, dailyLimit, exhausted, remaining };
}

/** Get usage history for the last N days. */
export async function getUsageHistory(agentId: string, days = 7): Promise<DailyTokenRecord[]> {
  const filePath = resolveBudgetPath(agentId);
  const data = await readBudgetFromPath(filePath);
  return data.daily.slice(-days);
}

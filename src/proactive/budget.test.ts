import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const testState = vi.hoisted(() => ({
  stateDir: "",
}));

vi.mock("../core/config/paths.js", () => ({
  resolveStateDir: () => testState.stateDir,
}));

import {
  getBudgetStatus,
  getTodayTokenUsage,
  getUsageHistory,
  recordTokenUsage,
} from "./budget.js";

let stateDir = "";

function agentId(label: string): string {
  return `test-${label}-${randomUUID().slice(0, 8)}`;
}

beforeEach(async () => {
  stateDir = path.join(os.tmpdir(), `marv-budget-test-${randomUUID()}`);
  testState.stateDir = stateDir;
  await fs.mkdir(stateDir, { recursive: true });
});

afterEach(async () => {
  await fs.rm(stateDir, { recursive: true, force: true });
});

describe("budget", () => {
  it("records and reads token usage", async () => {
    const id = agentId("rec");
    await recordTokenUsage(id, 1000);
    await recordTokenUsage(id, 500);
    const usage = await getTodayTokenUsage(id);
    expect(usage).toBe(1500);
  });

  it("skips zero or negative token amounts", async () => {
    const id = agentId("skip");
    await recordTokenUsage(id, 0);
    await recordTokenUsage(id, -100);
    const usage = await getTodayTokenUsage(id);
    expect(usage).toBe(0);
  });

  it("returns exhausted when over budget", async () => {
    const id = agentId("exhaust");
    await recordTokenUsage(id, 5000);
    const status = await getBudgetStatus(id, 3000);
    expect(status.exhausted).toBe(true);
    expect(status.remaining).toBe(0);
    expect(status.todayTokens).toBe(5000);
  });

  it("returns not exhausted when under budget", async () => {
    const id = agentId("under");
    await recordTokenUsage(id, 1000);
    const status = await getBudgetStatus(id, 5000);
    expect(status.exhausted).toBe(false);
    expect(status.remaining).toBe(4000);
  });

  it("treats 0 limit as unlimited", async () => {
    const id = agentId("unlim");
    await recordTokenUsage(id, 999_999);
    const status = await getBudgetStatus(id, 0);
    expect(status.exhausted).toBe(false);
    expect(status.remaining).toBe(0);
  });

  it("returns usage history", async () => {
    const id = agentId("hist");
    await recordTokenUsage(id, 100);
    const history = await getUsageHistory(id, 7);
    expect(history.length).toBe(1);
    expect(history[0].tokens).toBe(100);
  });

  it("returns empty for fresh agent", async () => {
    const id = agentId("fresh");
    const usage = await getTodayTokenUsage(id);
    expect(usage).toBe(0);
    const status = await getBudgetStatus(id, 10000);
    expect(status.exhausted).toBe(false);
    expect(status.remaining).toBe(10000);
  });
});

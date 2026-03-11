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
  appendToDigestBuffer,
  clearDeliveredEntries,
  flushDigestBuffer,
  readDigestBuffer,
} from "./digest-buffer.js";

let stateDir = "";

function createAgentId(label: string): string {
  return `${label}-${randomUUID()}`;
}

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-proactive-"));
  testState.stateDir = stateDir;
});

afterEach(async () => {
  testState.stateDir = "";
  await fs.rm(stateDir, { recursive: true, force: true });
});

describe("digest buffer", () => {
  it("stores and flushes pending entries", async () => {
    const agentId = createAgentId("store-flush");

    await appendToDigestBuffer(agentId, {
      source: "github",
      summary: "new issue",
      urgency: "normal",
    });

    const beforeFlush = await readDigestBuffer(agentId);
    expect(beforeFlush.entries).toHaveLength(1);
    expect(beforeFlush.entries[0]?.delivered).toBe(false);

    const flushed = await flushDigestBuffer(agentId);
    expect(flushed).toHaveLength(1);
    expect(flushed[0]?.summary).toBe("new issue");

    const afterFlush = await readDigestBuffer(agentId);
    expect(afterFlush.entries[0]?.delivered).toBe(true);
  });

  it("preserves all entries across parallel appends", async () => {
    const agentId = createAgentId("parallel-append");

    await Promise.all(
      Array.from({ length: 12 }, (_, index) =>
        appendToDigestBuffer(agentId, {
          source: "github",
          summary: `issue-${index}`,
          urgency: index % 2 === 0 ? "normal" : "urgent",
        }),
      ),
    );

    const buffer = await readDigestBuffer(agentId);
    expect(buffer.entries).toHaveLength(12);
    expect(new Set(buffer.entries.map((entry) => entry.summary))).toEqual(
      new Set(Array.from({ length: 12 }, (_, index) => `issue-${index}`)),
    );
    expect(buffer.entries.every((entry) => !entry.delivered)).toBe(true);
  });

  it("serializes concurrent flushes so entries are delivered once", async () => {
    const agentId = createAgentId("parallel-flush");

    await appendToDigestBuffer(agentId, {
      source: "github",
      summary: "single issue",
      urgency: "normal",
    });

    const [firstFlush, secondFlush] = await Promise.all([
      flushDigestBuffer(agentId),
      flushDigestBuffer(agentId),
    ]);

    expect(firstFlush.length + secondFlush.length).toBe(1);
    expect([firstFlush[0]?.summary, secondFlush[0]?.summary].filter(Boolean)).toEqual([
      "single issue",
    ]);

    const afterFlush = await readDigestBuffer(agentId);
    expect(afterFlush.entries).toHaveLength(1);
    expect(afterFlush.entries[0]?.delivered).toBe(true);
  });

  it("clears delivered entries without touching pending ones", async () => {
    const agentId = createAgentId("clear-delivered");

    await appendToDigestBuffer(agentId, {
      source: "github",
      summary: "delivered",
      urgency: "normal",
    });
    await appendToDigestBuffer(agentId, {
      source: "github",
      summary: "pending",
      urgency: "urgent",
    });

    const flushed = await flushDigestBuffer(agentId);
    expect(flushed).toHaveLength(2);

    await appendToDigestBuffer(agentId, {
      source: "github",
      summary: "new pending",
      urgency: "normal",
    });
    await clearDeliveredEntries(agentId);

    const buffer = await readDigestBuffer(agentId);
    expect(buffer.entries).toHaveLength(1);
    expect(buffer.entries[0]?.summary).toBe("new pending");
    expect(buffer.entries[0]?.delivered).toBe(false);
  });
});

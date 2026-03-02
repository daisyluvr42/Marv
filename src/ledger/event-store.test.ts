import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { appendLedgerEvent, queryLedgerEvents } from "./event-store.js";

let stateDir = "";
let prevStateDir: string | undefined;

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-ledger-"));
  prevStateDir = process.env.MARV_STATE_DIR;
  process.env.MARV_STATE_DIR = stateDir;
});

afterEach(async () => {
  if (prevStateDir == null) {
    delete process.env.MARV_STATE_DIR;
  } else {
    process.env.MARV_STATE_DIR = prevStateDir;
  }
  if (stateDir) {
    await fs.rm(stateDir, { recursive: true, force: true });
  }
});

describe("ledger event store", () => {
  it("appends and queries events in timestamp order", () => {
    appendLedgerEvent({
      eventId: "evt_a",
      conversationId: "conv:test",
      type: "InputEvent",
      ts: 100,
      payload: { message: "a" },
    });
    appendLedgerEvent({
      eventId: "evt_b",
      conversationId: "conv:test",
      taskId: "task-1",
      type: "CompletionEvent",
      ts: 200,
      payload: { message: "b" },
    });

    const all = queryLedgerEvents({ conversationId: "conv:test" });
    expect(all.map((entry) => entry.eventId)).toEqual(["evt_a", "evt_b"]);

    const filtered = queryLedgerEvents({
      conversationId: "conv:test",
      taskId: "task-1",
      type: "CompletionEvent",
    });
    expect(filtered).toHaveLength(1);
    expect(filtered[0]?.eventId).toBe("evt_b");
  });

  it("applies time range and limit filters", () => {
    appendLedgerEvent({
      conversationId: "conv:test",
      type: "TypeA",
      ts: 100,
      payload: { index: 1 },
    });
    appendLedgerEvent({
      conversationId: "conv:test",
      type: "TypeA",
      ts: 200,
      payload: { index: 2 },
    });
    appendLedgerEvent({
      conversationId: "conv:test",
      type: "TypeA",
      ts: 300,
      payload: { index: 3 },
    });

    const ranged = queryLedgerEvents({
      conversationId: "conv:test",
      fromTs: 150,
      toTs: 280,
      limit: 10,
    });
    expect(ranged).toHaveLength(1);
    expect(ranged[0]?.ts).toBe(200);

    const limited = queryLedgerEvents({
      conversationId: "conv:test",
      limit: 2,
    });
    expect(limited).toHaveLength(2);
  });
});

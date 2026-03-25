import { describe, expect, it } from "vitest";
import { isHeartbeatRun, resolveReplyAckBehavior, resolveReplyRunMode } from "./types.js";

describe("auto-reply run mode helpers", () => {
  it("prefers explicit runMode over legacy heartbeat flag", () => {
    expect(
      resolveReplyRunMode({
        isHeartbeat: false,
        runMode: {
          kind: "heartbeat",
          reason: "manual",
          ackToken: "HEARTBEAT_OK",
          maxAckChars: 120,
          visibility: "broadcast",
        },
      }),
    ).toEqual({
      kind: "heartbeat",
      reason: "manual",
      ackToken: "HEARTBEAT_OK",
      maxAckChars: 120,
      visibility: "broadcast",
    });
  });

  it("keeps legacy heartbeat runs working when runMode is absent", () => {
    expect(isHeartbeatRun({ isHeartbeat: true })).toBe(true);
    expect(resolveReplyRunMode({ isHeartbeat: true })).toEqual({
      kind: "heartbeat",
      reason: "other",
      ackToken: "HEARTBEAT_OK",
      maxAckChars: 300,
      visibility: "hidden",
    });
  });

  it("defaults to user mode when no special flags are present", () => {
    expect(isHeartbeatRun({})).toBe(false);
    expect(resolveReplyRunMode(undefined)).toEqual({ kind: "user" });
  });

  it("derives ack behavior from explicit heartbeat run modes", () => {
    expect(
      resolveReplyAckBehavior({
        runMode: {
          kind: "heartbeat",
          reason: "cron",
          ackToken: "MAINT_OK",
          maxAckChars: 42,
          visibility: "log",
        },
      }),
    ).toEqual({
      token: "MAINT_OK",
      mode: "heartbeat",
      maxAckChars: 42,
    });
  });

  it("returns no ack behavior for ordinary user runs", () => {
    expect(resolveReplyAckBehavior(undefined)).toBeNull();
  });
});

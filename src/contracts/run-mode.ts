import type { HeartbeatReasonKind } from "../infra/heartbeat/heartbeat-reason.js";

export type SpecialRunMode =
  | { kind: "user" }
  | {
      kind: "heartbeat";
      reason: HeartbeatReasonKind;
      ackToken: string;
      maxAckChars: number;
      visibility: "hidden" | "log" | "broadcast";
    }
  | { kind: "followup" }
  | { kind: "proactive"; trigger: string }
  | { kind: "tool-reentry" }
  | { kind: "maintenance" };

export type { HeartbeatReasonKind } from "../infra/heartbeat/heartbeat-reason.js";

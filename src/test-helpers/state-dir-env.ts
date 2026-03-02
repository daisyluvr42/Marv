import { captureEnv } from "../test-utils/env.js";

export function snapshotStateDirEnv() {
  return captureEnv(["MARV_STATE_DIR", "CLAWDBOT_STATE_DIR"]);
}

export function restoreStateDirEnv(snapshot: ReturnType<typeof snapshotStateDirEnv>): void {
  snapshot.restore();
}

export function setStateDirEnv(stateDir: string): void {
  process.env.MARV_STATE_DIR = stateDir;
  delete process.env.CLAWDBOT_STATE_DIR;
}

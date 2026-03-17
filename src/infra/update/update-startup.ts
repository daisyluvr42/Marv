import fs from "node:fs/promises";
import path from "node:path";
import type { loadConfig } from "../../core/config/config.js";
import { resolveStateDir } from "../../core/config/paths.js";
import {
  checkForUpdate,
  DEFAULT_UPDATE_CHECK_INTERVAL_MS,
  formatAvailableUpdateSummary,
  shouldNotifyForVersion,
} from "./update-notify.js";
import { checkVersionDrift } from "./version-drift.js";

type UpdateCheckState = {
  lastCheckedAt?: string;
  lastNotifiedVersion?: string;
  lastNotifiedTag?: string;
};

const UPDATE_CHECK_FILENAME = "update-check.json";

function shouldSkipCheck(allowInTests: boolean): boolean {
  if (allowInTests) {
    return false;
  }
  if (process.env.VITEST || process.env.NODE_ENV === "test") {
    return true;
  }
  return false;
}

async function readState(statePath: string): Promise<UpdateCheckState> {
  try {
    const raw = await fs.readFile(statePath, "utf-8");
    const parsed = JSON.parse(raw) as UpdateCheckState;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

async function writeState(statePath: string, state: UpdateCheckState): Promise<void> {
  await fs.mkdir(path.dirname(statePath), { recursive: true });
  await fs.writeFile(statePath, JSON.stringify(state, null, 2), "utf-8");
}

export async function runGatewayUpdateCheck(params: {
  cfg: ReturnType<typeof loadConfig>;
  log: { info: (msg: string, meta?: Record<string, unknown>) => void };
  isNixMode: boolean;
  allowInTests?: boolean;
}): Promise<void> {
  if (shouldSkipCheck(Boolean(params.allowInTests))) {
    return;
  }
  if (params.isNixMode) {
    return;
  }
  if (params.cfg.update?.checkOnStart === false) {
    return;
  }

  const statePath = path.join(resolveStateDir(), UPDATE_CHECK_FILENAME);
  const state = await readState(statePath);
  const now = Date.now();
  const lastCheckedAt = state.lastCheckedAt ? Date.parse(state.lastCheckedAt) : null;
  if (lastCheckedAt && Number.isFinite(lastCheckedAt)) {
    if (now - lastCheckedAt < DEFAULT_UPDATE_CHECK_INTERVAL_MS) {
      return;
    }
  }

  const update = await checkForUpdate({
    cfg: params.cfg,
    timeoutMs: 2500,
    fetchGit: false,
  });

  const nextState: UpdateCheckState = {
    ...state,
    lastCheckedAt: new Date(now).toISOString(),
  };

  if (update.installKind !== "package") {
    await writeState(statePath, nextState);
    return;
  }

  if (shouldNotifyForVersion({ update, ...state })) {
    params.log.info(formatAvailableUpdateSummary(update));
    nextState.lastNotifiedVersion = update.latestVersion ?? undefined;
    nextState.lastNotifiedTag = update.tag;
  }

  await writeState(statePath, nextState);

  // Check for macOS App ↔ CLI version drift (non-blocking)
  const drift = await checkVersionDrift().catch(() => null);
  if (drift?.drifted && drift.message) {
    params.log.info(drift.message);
  }
}

export function scheduleGatewayUpdateCheck(params: {
  cfg: ReturnType<typeof loadConfig>;
  log: { info: (msg: string, meta?: Record<string, unknown>) => void };
  isNixMode: boolean;
}): void {
  void runGatewayUpdateCheck(params).catch(() => {});
}

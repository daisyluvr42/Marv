import { formatCliCommand } from "../../cli/command-format.js";
import type { loadConfig } from "../../core/config/config.js";
import { VERSION } from "../../version.js";
import { resolveMarvPackageRoot } from "../marv-root.js";
import { resolveEffectiveUpdateChannel, type UpdateChannel } from "./update-channels.js";
import {
  checkUpdateStatus,
  compareSemverStrings,
  type GitUpdateStatus,
  resolveNpmChannelTag,
} from "./update-check.js";

export const DEFAULT_UPDATE_CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000;

export type UpdateCheckNotification = {
  available: boolean;
  currentVersion: string;
  latestVersion: string | null;
  channel: UpdateChannel;
  tag: string;
  installKind: "git" | "package" | "unknown";
  git?: {
    branch: string | null;
    currentTag: string | null;
    currentSha: string | null;
    upstream: string | null;
    upstreamSha: string | null;
    behind: number | null;
    approval?: GitUpdateStatus["approval"];
  };
};

function shortSha(value: string | null | undefined): string | null {
  const trimmed = value?.trim();
  return trimmed ? trimmed.slice(0, 8) : null;
}

function formatCurrentGitVersion(git?: GitUpdateStatus): string {
  return shortSha(git?.sha) ?? git?.tag?.trim() ?? "unknown";
}

function formatLatestGitVersion(git?: GitUpdateStatus): string | null {
  if (git?.approval?.approvedSha) {
    return shortSha(git.approval.approvedSha) ?? git.approval.approvedTag ?? null;
  }
  return (
    shortSha(git?.upstreamSha) ??
    git?.upstream?.trim() ??
    (typeof git?.behind === "number" && git.behind > 0 ? `upstream+${git.behind}` : null)
  );
}

export async function checkForUpdate(params: {
  cfg?: ReturnType<typeof loadConfig>;
  channel?: UpdateChannel | null;
  timeoutMs?: number;
  root?: string | null;
  fetchGit?: boolean;
}): Promise<UpdateCheckNotification> {
  const timeoutMs = params.timeoutMs ?? 2_500;
  const root =
    params.root ??
    (await resolveMarvPackageRoot({
      moduleUrl: import.meta.url,
      argv1: process.argv[1],
      cwd: process.cwd(),
    }));
  const status = await checkUpdateStatus({
    root,
    timeoutMs,
    fetchGit: params.fetchGit ?? true,
    includeRegistry: false,
    approval: params.cfg?.update?.approval,
  });

  const configuredChannel = params.channel ?? params.cfg?.update?.channel ?? null;
  const channelInfo = resolveEffectiveUpdateChannel({
    configChannel: configuredChannel,
    installKind: status.installKind,
    git: status.git,
  });

  if (status.installKind === "git") {
    const approvedSha = status.git?.approval?.approvedSha ?? null;
    const approvalRequired = status.git?.approval?.required === true;
    const behind = typeof status.git?.behind === "number" ? status.git.behind : null;
    const available = approvalRequired
      ? Boolean(approvedSha && status.git?.sha && approvedSha !== status.git.sha)
      : behind != null && behind > 0;
    return {
      available,
      currentVersion: formatCurrentGitVersion(status.git),
      latestVersion: formatLatestGitVersion(status.git),
      channel: channelInfo.channel,
      tag: status.git?.approval?.approvedTag ?? channelInfo.channel,
      installKind: "git",
      git: {
        branch: status.git?.branch ?? null,
        currentTag: status.git?.tag ?? null,
        currentSha: status.git?.sha ?? null,
        upstream: status.git?.upstream ?? null,
        upstreamSha: status.git?.upstreamSha ?? null,
        behind,
        approval: status.git?.approval ?? null,
      },
    };
  }

  const resolved = await resolveNpmChannelTag({
    channel: channelInfo.channel,
    timeoutMs,
  });
  const latestVersion = resolved.version;
  const cmp = compareSemverStrings(VERSION, latestVersion);

  return {
    available: cmp != null && cmp < 0,
    currentVersion: VERSION,
    latestVersion,
    channel: channelInfo.channel,
    tag: resolved.tag,
    installKind: status.installKind,
  };
}

export function formatAvailableUpdateSummary(update: UpdateCheckNotification): string {
  if (update.installKind === "git") {
    const channelLabel = `${update.channel} channel`;
    const latest = update.latestVersion ?? "upstream";
    const approvalLabel = update.git?.approval?.approvedTag
      ? ` approved by ${update.git.approval.approvedTag}`
      : update.git?.approval?.required
        ? " approved deploy"
        : "";
    return `Marv update available (${channelLabel}${approvalLabel}): ${update.currentVersion} -> ${latest}. Run ${formatCliCommand("marv update")}.`;
  }
  return `Marv v${update.latestVersion ?? "unknown"} is available (current v${update.currentVersion}). Run ${formatCliCommand("marv update")}.`;
}

export function buildUpdateNotificationPrompt(update: UpdateCheckNotification): string {
  if (update.installKind === "git") {
    const branch = update.git?.branch ? ` on ${update.git.branch}` : "";
    const behind =
      typeof update.git?.behind === "number" && update.git.behind > 0
        ? ` The checkout is ${update.git.behind} commit${update.git.behind === 1 ? "" : "s"} behind.`
        : "";
    const approval =
      update.git?.approval?.required === true
        ? update.git.approval.approvedTag
          ? ` Approved deploy tag: ${update.git.approval.approvedTag}.`
          : " Deploy approval is required before git updates can move this checkout."
        : "";
    const latest = update.latestVersion
      ? update.git?.approval?.required
        ? `Latest approved target: ${update.latestVersion}.`
        : `Latest upstream commit: ${update.latestVersion}.`
      : "";
    return [
      "Notify the user that a Marv update is available.",
      `Install type: git${branch}.`,
      `Current commit: ${update.currentVersion}.`,
      latest,
      approval,
      `Channel: ${update.channel}.`,
      `${behind} Tell them they can run ${formatCliCommand("marv update")} or ask you to update it for them.`,
    ]
      .filter(Boolean)
      .join(" ");
  }

  return [
    "Notify the user that a Marv update is available.",
    `Current version: v${update.currentVersion}.`,
    update.latestVersion ? `Latest version: v${update.latestVersion}.` : "",
    `Channel/tag: ${update.channel}/${update.tag}.`,
    `Tell them they can run ${formatCliCommand("marv update")} or ask you to update it for them.`,
  ]
    .filter(Boolean)
    .join(" ");
}

export function resolveUpdateCheckEnabled(cfg: ReturnType<typeof loadConfig>): boolean {
  return cfg.update?.checkOnStart !== false;
}

export function resolveUpdateCheckIntervalMs(cfg: ReturnType<typeof loadConfig>): number {
  const intervalMs = cfg.update?.autoCheckIntervalMs;
  if (typeof intervalMs === "number" && Number.isFinite(intervalMs) && intervalMs > 0) {
    return Math.max(1_000, Math.floor(intervalMs));
  }
  return DEFAULT_UPDATE_CHECK_INTERVAL_MS;
}

export function shouldNotifyForVersion(params: {
  update: UpdateCheckNotification;
  lastNotifiedVersion?: string;
  lastNotifiedTag?: string;
}): boolean {
  if (!params.update.available || !params.update.latestVersion) {
    return false;
  }
  return (
    params.lastNotifiedVersion !== params.update.latestVersion ||
    params.lastNotifiedTag !== params.update.tag
  );
}

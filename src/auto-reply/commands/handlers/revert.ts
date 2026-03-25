import { logVerbose } from "../../../globals.js";
import {
  listWorkspaceSnapshots,
  revertWorkspaceToSnapshot,
} from "../../../infra/workspace-rollback.js";
import type { CommandRegistry } from "../command-registry.js";
import type { CommandHandler } from "../handler-types.js";

const MAX_LIST_ITEMS = 5;

function formatSnapshotLine(index: number, ts: number, commit: string, mode: string): string {
  const shortCommit = commit.length > 12 ? commit.slice(0, 12) : commit;
  return `${index}. ${new Date(ts).toISOString()}  ${shortCommit}  (${mode})`;
}

export const handleRevertCommand: CommandHandler = async (params, allowTextCommands) => {
  if (!allowTextCommands) {
    return null;
  }

  const normalized = params.command.commandBodyNormalized;
  if (normalized !== "/revert" && !normalized.startsWith("/revert ")) {
    return null;
  }
  if (!params.command.isAuthorizedSender) {
    logVerbose(
      `Ignoring /revert from unauthorized sender: ${params.command.senderId || "<unknown>"}`,
    );
    return { shouldContinue: false };
  }

  const rawArg = normalized === "/revert" ? "" : normalized.slice("/revert".length).trim();
  if (rawArg === "list") {
    const snapshots = await listWorkspaceSnapshots({
      workspaceDir: params.workspaceDir,
      limit: MAX_LIST_ITEMS,
    });
    if (snapshots.length === 0) {
      return {
        shouldContinue: false,
        reply: {
          text: "⚠️ No auto-snapshots found yet. A snapshot is created before write/edit/apply_patch.",
        },
      };
    }
    const lines = ["🕘 Recent auto-snapshots", ""];
    snapshots.forEach((snapshot, index) => {
      lines.push(formatSnapshotLine(index + 1, snapshot.ts, snapshot.commit, snapshot.mode));
    });
    lines.push("");
    lines.push("Usage: /revert (latest), /revert <index>, /revert <commit>");
    return {
      shouldContinue: false,
      reply: { text: lines.join("\n") },
    };
  }

  let snapshotRef: string | undefined;
  if (rawArg) {
    const asIndex = Number.parseInt(rawArg, 10);
    if (Number.isFinite(asIndex) && `${asIndex}` === rawArg && asIndex > 0) {
      const snapshots = await listWorkspaceSnapshots({
        workspaceDir: params.workspaceDir,
        limit: asIndex,
      });
      const selected = snapshots[asIndex - 1];
      if (!selected) {
        return {
          shouldContinue: false,
          reply: {
            text: `⚠️ Snapshot #${asIndex} not found. Use /revert list to inspect available snapshots.`,
          },
        };
      }
      snapshotRef = selected.commit;
    } else {
      snapshotRef = rawArg;
    }
  }

  const reverted = await revertWorkspaceToSnapshot({
    workspaceDir: params.workspaceDir,
    snapshotRef,
  });
  if (!reverted.ok) {
    const text =
      reverted.code === "not_git_repo"
        ? "⚠️ /revert is only available inside a git repository workspace."
        : reverted.code === "snapshot_not_found"
          ? "⚠️ No auto-snapshot is available yet. Run some write/edit/apply_patch actions first."
          : reverted.code === "invalid_ref"
            ? `⚠️ Snapshot reference not found: ${snapshotRef}.`
            : reverted.message;
    return {
      shouldContinue: false,
      reply: { text },
    };
  }

  const commit =
    reverted.snapshot.commit.length > 12
      ? reverted.snapshot.commit.slice(0, 12)
      : reverted.snapshot.commit;
  const cleanSuffix = reverted.cleaned ? "" : " (reset completed; clean step partially failed)";
  return {
    shouldContinue: false,
    reply: {
      text: `✅ Reverted workspace to ${commit} from ${new Date(reverted.snapshot.ts).toISOString()}${cleanSuffix}.`,
    },
  };
};

export function registerRevertCommands(registry: CommandRegistry): void {
  registry.register("revert", handleRevertCommand);
}

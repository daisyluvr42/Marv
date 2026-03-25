import fs from "node:fs/promises";
import path from "node:path";
import { Type } from "@sinclair/typebox";
import { loadConfig, writeConfigFile } from "../../../core/config/config.js";
import { resolveAgentIdFromSessionKey } from "../../../routing/session-key.js";
import { resolveAgentWorkspaceDir } from "../../agent-scope.js";
import { DEFAULT_HEARTBEAT_FILENAME } from "../../workspace.js";
import type { AnyAgentTool } from "../common.js";
import { readNumberParam, readStringParam, ToolInputError } from "../common.js";
import {
  applyOptionalStringPatch,
  buildGenericDeniedResult,
  buildInvalidResult,
  normalizeHeartbeatFileAction,
  normalizePatchString,
  pruneEmptyObject,
  readBooleanParam,
  type HeartbeatFileAction,
  type SelfSettingsArgs,
  type SelfSettingsToolOpts,
} from "./self-settings-normalize.js";

const HeartbeatSettingsToolSchema = Type.Object(
  {
    heartbeatEvery: Type.Optional(Type.String()),
    heartbeatPrompt: Type.Optional(Type.String()),
    heartbeatModel: Type.Optional(Type.String()),
    heartbeatTarget: Type.Optional(Type.String()),
    heartbeatTo: Type.Optional(Type.String()),
    heartbeatAccountId: Type.Optional(Type.String()),
    heartbeatIncludeReasoning: Type.Optional(Type.Boolean()),
    heartbeatSuppressToolErrorWarnings: Type.Optional(Type.Boolean()),
    heartbeatAckMaxChars: Type.Optional(Type.Number({ minimum: 0 })),
    heartbeatActiveHoursStart: Type.Optional(Type.String()),
    heartbeatActiveHoursEnd: Type.Optional(Type.String()),
    heartbeatActiveHoursTimezone: Type.Optional(Type.String()),
    heartbeatFileAction: Type.Optional(Type.String()),
    heartbeatFileContent: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export function createHeartbeatSettingsTool(opts?: SelfSettingsToolOpts): AnyAgentTool {
  return {
    label: "Heartbeat Settings",
    name: "self_settings_heartbeat",
    description:
      "Apply shared heartbeat settings: schedule, prompt, model, target, active hours, ack limits, and HEARTBEAT.md file operations (replace, append, clear).",
    parameters: HeartbeatSettingsToolSchema,
    execute: async (_toolCallId, args) => {
      if (!opts?.agentSessionKey?.trim()) {
        throw new ToolInputError("sessionKey required");
      }

      const params = args as SelfSettingsArgs;
      const cfg = opts.config ?? loadConfig();
      const agentId = resolveAgentIdFromSessionKey(opts.agentSessionKey.trim());

      const heartbeatEvery = normalizePatchString(readStringParam(params, "heartbeatEvery"));
      const heartbeatPrompt = normalizePatchString(readStringParam(params, "heartbeatPrompt"));
      const heartbeatModel = normalizePatchString(readStringParam(params, "heartbeatModel"));
      const heartbeatTarget = normalizePatchString(readStringParam(params, "heartbeatTarget"));
      const heartbeatTo = normalizePatchString(readStringParam(params, "heartbeatTo"));
      const heartbeatAccountId = normalizePatchString(
        readStringParam(params, "heartbeatAccountId"),
      );
      const heartbeatIncludeReasoning = readBooleanParam(params, "heartbeatIncludeReasoning");
      const heartbeatSuppressToolErrorWarnings = readBooleanParam(
        params,
        "heartbeatSuppressToolErrorWarnings",
      );
      const heartbeatAckMaxChars = readNumberParam(params, "heartbeatAckMaxChars");
      const heartbeatActiveHoursStart = normalizePatchString(
        readStringParam(params, "heartbeatActiveHoursStart"),
      );
      const heartbeatActiveHoursEnd = normalizePatchString(
        readStringParam(params, "heartbeatActiveHoursEnd"),
      );
      const heartbeatActiveHoursTimezone = normalizePatchString(
        readStringParam(params, "heartbeatActiveHoursTimezone"),
      );
      const heartbeatFileActionRaw = readStringParam(params, "heartbeatFileAction");
      const heartbeatFileAction = normalizeHeartbeatFileAction(heartbeatFileActionRaw);
      const heartbeatFileContent = readStringParam(params, "heartbeatFileContent", {
        trim: false,
        allowEmpty: true,
      });

      const hasHeartbeatConfigChange =
        heartbeatEvery !== undefined ||
        heartbeatPrompt !== undefined ||
        heartbeatModel !== undefined ||
        heartbeatTarget !== undefined ||
        heartbeatTo !== undefined ||
        heartbeatAccountId !== undefined ||
        heartbeatIncludeReasoning !== undefined ||
        heartbeatSuppressToolErrorWarnings !== undefined ||
        heartbeatAckMaxChars !== undefined ||
        heartbeatActiveHoursStart !== undefined ||
        heartbeatActiveHoursEnd !== undefined ||
        heartbeatActiveHoursTimezone !== undefined;
      const hasHeartbeatFileChange = heartbeatFileActionRaw !== undefined;

      if (!hasHeartbeatConfigChange && !hasHeartbeatFileChange) {
        throw new ToolInputError("at least one setting change is required");
      }

      if (opts.directUserInstruction === false) {
        return buildGenericDeniedResult();
      }

      // --- Validation ---
      if (heartbeatFileActionRaw !== undefined && heartbeatFileAction === undefined) {
        return buildInvalidResult();
      }
      if (
        heartbeatFileAction &&
        heartbeatFileAction !== "clear" &&
        heartbeatFileContent === undefined
      ) {
        return buildInvalidResult();
      }
      if (heartbeatAckMaxChars !== undefined && !Number.isInteger(heartbeatAckMaxChars)) {
        return buildInvalidResult();
      }

      // --- Apply config changes ---
      const sharedHeartbeatLabels: string[] = [];
      let nextConfig = cfg;

      if (hasHeartbeatConfigChange) {
        const nextActiveHours = {
          ...nextConfig.agents?.defaults?.heartbeat?.activeHours,
        } as Record<string, unknown>;
        applyOptionalStringPatch(nextActiveHours, "start", heartbeatActiveHoursStart);
        applyOptionalStringPatch(nextActiveHours, "end", heartbeatActiveHoursEnd);
        applyOptionalStringPatch(nextActiveHours, "timezone", heartbeatActiveHoursTimezone);

        const nextHeartbeat = {
          ...nextConfig.agents?.defaults?.heartbeat,
        } as Record<string, unknown>;
        applyOptionalStringPatch(nextHeartbeat, "every", heartbeatEvery);
        applyOptionalStringPatch(nextHeartbeat, "prompt", heartbeatPrompt);
        applyOptionalStringPatch(nextHeartbeat, "model", heartbeatModel);
        applyOptionalStringPatch(nextHeartbeat, "target", heartbeatTarget);
        applyOptionalStringPatch(nextHeartbeat, "to", heartbeatTo);
        applyOptionalStringPatch(nextHeartbeat, "accountId", heartbeatAccountId);
        if (heartbeatIncludeReasoning !== undefined) {
          nextHeartbeat.includeReasoning = heartbeatIncludeReasoning;
        }
        if (heartbeatSuppressToolErrorWarnings !== undefined) {
          nextHeartbeat.suppressToolErrorWarnings = heartbeatSuppressToolErrorWarnings;
        }
        if (heartbeatAckMaxChars !== undefined) {
          nextHeartbeat.ackMaxChars = Math.max(0, Math.trunc(heartbeatAckMaxChars));
        }
        const prunedActiveHours = pruneEmptyObject(nextActiveHours);
        if (prunedActiveHours) {
          nextHeartbeat.activeHours = prunedActiveHours;
        } else {
          delete nextHeartbeat.activeHours;
        }

        nextConfig = {
          ...nextConfig,
          agents: {
            ...nextConfig.agents,
            defaults: {
              ...nextConfig.agents?.defaults,
              heartbeat: nextHeartbeat,
            },
          },
        };

        try {
          await writeConfigFile(nextConfig);
        } catch {
          return buildInvalidResult();
        }

        if (heartbeatEvery !== undefined) {
          sharedHeartbeatLabels.push(`heartbeat every ${heartbeatEvery ?? "default"}`);
        }
        if (heartbeatPrompt !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat prompt ${heartbeatPrompt === null ? "default" : "configured"}`,
          );
        }
        if (heartbeatModel !== undefined) {
          sharedHeartbeatLabels.push(`heartbeat model ${heartbeatModel ?? "default"}`);
        }
        if (heartbeatTarget !== undefined) {
          sharedHeartbeatLabels.push(`heartbeat target ${heartbeatTarget ?? "default"}`);
        }
        if (heartbeatTo !== undefined) {
          sharedHeartbeatLabels.push(`heartbeat recipient ${heartbeatTo ?? "default"}`);
        }
        if (heartbeatAccountId !== undefined) {
          sharedHeartbeatLabels.push(`heartbeat account ${heartbeatAccountId ?? "default"}`);
        }
        if (heartbeatIncludeReasoning !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat reasoning ${heartbeatIncludeReasoning ? "enabled" : "disabled"}`,
          );
        }
        if (heartbeatSuppressToolErrorWarnings !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat tool warnings ${heartbeatSuppressToolErrorWarnings ? "suppressed" : "default"}`,
          );
        }
        if (heartbeatAckMaxChars !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat ack max chars ${Math.max(0, Math.trunc(heartbeatAckMaxChars))}`,
          );
        }
        if (heartbeatActiveHoursStart !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat active start ${heartbeatActiveHoursStart ?? "default"}`,
          );
        }
        if (heartbeatActiveHoursEnd !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat active end ${heartbeatActiveHoursEnd ?? "default"}`,
          );
        }
        if (heartbeatActiveHoursTimezone !== undefined) {
          sharedHeartbeatLabels.push(
            `heartbeat active timezone ${heartbeatActiveHoursTimezone ?? "default"}`,
          );
        }
      }

      // --- Heartbeat file operations ---
      let heartbeatFileSummary: string | undefined;
      let heartbeatFileDetails:
        | {
            action: HeartbeatFileAction;
            path: string;
            size: number;
          }
        | undefined;
      if (heartbeatFileAction) {
        try {
          const workspaceDir = resolveAgentWorkspaceDir(nextConfig, agentId);
          await fs.mkdir(workspaceDir, { recursive: true });
          const heartbeatPath = path.join(workspaceDir, DEFAULT_HEARTBEAT_FILENAME);
          let nextContent = "";
          if (heartbeatFileAction === "replace") {
            nextContent = heartbeatFileContent ?? "";
          } else if (heartbeatFileAction === "append") {
            const current = await fs.readFile(heartbeatPath, "utf-8").catch(() => "");
            const separator = current.length > 0 && !current.endsWith("\n") ? "\n" : "";
            nextContent = `${current}${separator}${heartbeatFileContent ?? ""}`;
          }
          await fs.writeFile(heartbeatPath, nextContent, "utf-8");
          heartbeatFileSummary = `Updated HEARTBEAT.md: ${heartbeatFileAction}.`;
          heartbeatFileDetails = {
            action: heartbeatFileAction,
            path: heartbeatPath,
            size: nextContent.length,
          };
        } catch {
          return buildInvalidResult();
        }
      }

      const summaryParts = [
        sharedHeartbeatLabels.length > 0
          ? `Updated shared heartbeat settings: ${sharedHeartbeatLabels.join("; ")}.`
          : null,
        heartbeatFileSummary ?? null,
      ].filter((value): value is string => Boolean(value));

      return {
        content: [
          {
            type: "text" as const,
            text:
              summaryParts.length > 0
                ? summaryParts.join(" ")
                : "No heartbeat setting changes were needed.",
          },
        ],
        details: {
          ok: true,
          applied: true,
          sharedConfig: hasHeartbeatConfigChange
            ? {
                heartbeatEvery: nextConfig.agents?.defaults?.heartbeat?.every,
                heartbeatPrompt: nextConfig.agents?.defaults?.heartbeat?.prompt,
                heartbeatModel: nextConfig.agents?.defaults?.heartbeat?.model,
                heartbeatTarget: nextConfig.agents?.defaults?.heartbeat?.target,
                heartbeatTo: nextConfig.agents?.defaults?.heartbeat?.to,
                heartbeatAccountId: nextConfig.agents?.defaults?.heartbeat?.accountId,
                heartbeatIncludeReasoning: nextConfig.agents?.defaults?.heartbeat?.includeReasoning,
                heartbeatSuppressToolErrorWarnings:
                  nextConfig.agents?.defaults?.heartbeat?.suppressToolErrorWarnings,
                heartbeatAckMaxChars: nextConfig.agents?.defaults?.heartbeat?.ackMaxChars,
                heartbeatActiveHoursStart:
                  nextConfig.agents?.defaults?.heartbeat?.activeHours?.start,
                heartbeatActiveHoursEnd: nextConfig.agents?.defaults?.heartbeat?.activeHours?.end,
                heartbeatActiveHoursTimezone:
                  nextConfig.agents?.defaults?.heartbeat?.activeHours?.timezone,
              }
            : undefined,
          files: heartbeatFileDetails
            ? {
                heartbeat: heartbeatFileDetails,
              }
            : undefined,
        },
      };
    },
  };
}

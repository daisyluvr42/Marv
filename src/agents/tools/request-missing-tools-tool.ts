import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../core/config/config.js";
import { resolveAgentIdFromSessionKey } from "../../routing/session-key.js";
import { readSkillUsageRecords } from "../skill-usage-records.js";
import {
  inspectDiscoveredSkillSafety,
  installDiscoveredSkill,
  type SkillInstallSafetyLevel,
} from "../skills-install.js";
import { resolveWorkspaceRoot } from "../workspace-dir.js";
import type { AnyAgentTool } from "./common.js";
import { jsonResult, readStringArrayParam, readStringParam } from "./common.js";
import { callGatewayTool, readGatewayCallOptions } from "./gateway.js";
import { ToolDiscoveryService, type MissingCapability } from "./tool-discovery.js";

const RequestMissingToolsSchema = Type.Object(
  {
    description: Type.String({ minLength: 1 }),
    suggestedTools: Type.Optional(Type.Array(Type.String())),
    contextTaskId: Type.Optional(Type.String()),
    autoInstall: Type.Optional(Type.Boolean()),
    installLimit: Type.Optional(Type.Integer({ minimum: 1, maximum: 5 })),
    gatewayUrl: Type.Optional(Type.String()),
    gatewayToken: Type.Optional(Type.String()),
    timeoutMs: Type.Optional(Type.Integer({ minimum: 1 })),
  },
  { additionalProperties: false },
);

type ApprovalDecision =
  | "allow-once"
  | "allow-always"
  | "deny"
  | "unavailable"
  | "not-needed"
  | "blocked";

const seenCapabilitySearches = new Map<string, Set<string>>();

export const __testing = {
  resetSeenCapabilitySearches() {
    seenCapabilitySearches.clear();
  },
} as const;

function normalizeApprovalDecision(raw: unknown): ApprovalDecision {
  if (raw === "allow-once" || raw === "allow-always" || raw === "deny") {
    return raw;
  }
  return "unavailable";
}

function resolveApprovalTimeoutMs(config?: MarvConfig): number {
  const configured = config?.autonomy?.escalation?.approvalTimeoutSeconds;
  const seconds = typeof configured === "number" && Number.isFinite(configured) ? configured : 120;
  return Math.max(1_000, Math.floor(seconds * 1000));
}

async function requestInstallApproval(params: {
  skillId: string;
  contextTaskId?: string;
  config?: MarvConfig;
  agentSessionKey?: string;
  gatewayOptions: ReturnType<typeof readGatewayCallOptions>;
}): Promise<ApprovalDecision> {
  try {
    const result = await callGatewayTool<{ decision?: string }>(
      "exec.approval.request",
      params.gatewayOptions,
      {
        command: `skills install ${params.skillId}`,
        kind: "skill-install",
        taskId: params.contextTaskId ?? null,
        ask: "always",
        agentId: params.agentSessionKey
          ? resolveAgentIdFromSessionKey(params.agentSessionKey)
          : null,
        sessionKey: params.agentSessionKey ?? null,
        timeoutMs: resolveApprovalTimeoutMs(params.config),
      },
      { expectFinal: true },
    );
    return normalizeApprovalDecision(result?.decision);
  } catch {
    return "unavailable";
  }
}

export function createRequestMissingToolsTool(opts?: {
  workspaceDir?: string;
  config?: MarvConfig;
  agentSessionKey?: string;
}): AnyAgentTool {
  return {
    label: "Tool Discovery",
    name: "request_missing_tools",
    description:
      "Discover skills for missing capabilities and optionally install approved matches.",
    parameters: RequestMissingToolsSchema,
    execute: async (_toolCallId, rawParams) => {
      const params =
        rawParams && typeof rawParams === "object" && !Array.isArray(rawParams)
          ? (rawParams as Record<string, unknown>)
          : {};
      const description = readStringParam(params, "description", {
        required: true,
        label: "description",
      });
      const suggestedTools = readStringArrayParam(params, "suggestedTools");
      const contextTaskId = readStringParam(params, "contextTaskId");
      const capability: MissingCapability = {
        description,
        ...(suggestedTools && suggestedTools.length > 0 ? { suggestedTools } : {}),
        ...(contextTaskId ? { contextTaskId } : {}),
      };
      const dedupeScope = contextTaskId?.trim() || opts?.agentSessionKey?.trim();
      const dedupeKey = description.trim().toLowerCase();
      if (dedupeScope && dedupeKey) {
        const seen = seenCapabilitySearches.get(dedupeScope) ?? new Set<string>();
        if (seen.has(dedupeKey)) {
          return jsonResult({
            ok: false,
            discovered: [],
            installed: [],
            message: "Discovery already attempted for this capability in the current task.",
          });
        }
        seen.add(dedupeKey);
        seenCapabilitySearches.set(dedupeScope, seen);
      }

      const discovery = new ToolDiscoveryService();
      const workspaceDir = resolveWorkspaceRoot(opts?.workspaceDir);
      const usageRecords = await readSkillUsageRecords();
      const discovered = discovery.discover({
        workspaceDir,
        capability,
        config: opts?.config,
        usageRecords,
        limit:
          typeof params.installLimit === "number" && Number.isFinite(params.installLimit)
            ? params.installLimit
            : 5,
      });

      if (discovered.length === 0) {
        const synthesisEnabled = opts?.config?.autonomy?.toolSynthesis?.enabled !== false;
        return jsonResult({
          ok: true,
          discovered: [],
          synthesisHint: synthesisEnabled
            ? {
                guidance:
                  "No existing skill matches. Create one using the appropriate path:\n" +
                  "• Managed CLI tool (recommended — persists across sessions and becomes discoverable):\n" +
                  "  (1) Write a wrapper script via `write` (Python or Bash) with stable arguments.\n" +
                  "  (2) Register with `cli_synthesize` (automatically creates a SKILL.md index entry).\n" +
                  "  (3) Run `cli_verify` to validate, then invoke via `cli_invoke`.\n" +
                  "• Lightweight one-off script (no external CLI, not needed across sessions):\n" +
                  "  `bun src/agents/tools/tool-synthesis.ts persist --name <name> --description <desc> --script <path>`",
              }
            : null,
          message: synthesisEnabled
            ? "No matching skills found. Consider creating an ad-hoc solution (see synthesisHint)."
            : "No matching skills found for the requested capability.",
        });
      }

      const autoInstallRequested = params.autoInstall !== false;
      const autoInstallEnabled = opts?.config?.autonomy?.autoInstallSkills !== false;
      if (!autoInstallEnabled) {
        return jsonResult({
          ok: true,
          discovered,
          installed: [],
          message: "Discovery completed. Auto-install is disabled by configuration.",
        });
      }

      const installApprovalMode = opts?.config?.autonomy?.discovery?.installApproval ?? "per-skill";
      const gatewayOptions = readGatewayCallOptions(params);
      const installCandidates =
        installApprovalMode === "batch" ? discovered.slice(0, 3) : discovered.slice(0, 1);

      const installed: Array<{
        skillId: string;
        approved: ApprovalDecision;
        ok: boolean;
        message: string;
        scanLevel?: SkillInstallSafetyLevel;
        warnings?: string[];
      }> = [];
      for (const candidate of installCandidates) {
        // Workspace/project skills are already locally present — no installation needed.
        if (candidate.alreadyInstalled) {
          installed.push({
            skillId: candidate.skillId,
            approved: "not-needed",
            ok: true,
            message: "Skill already available (workspace source; no install required).",
          });
          if (installApprovalMode !== "batch") {
            break;
          }
          continue;
        }

        const scan = await inspectDiscoveredSkillSafety({
          workspaceDir,
          skillId: candidate.skillId,
          config: opts?.config,
        });
        const scanLevel = scan?.level ?? "warn";
        const scanWarnings = scan?.warnings ?? [];
        if (scan?.blocked) {
          installed.push({
            skillId: candidate.skillId,
            approved: "blocked",
            ok: false,
            message: scanWarnings[0] ?? "Installation blocked by the safety scan.",
            scanLevel,
            warnings: scanWarnings,
          });
          if (installApprovalMode !== "batch") {
            break;
          }
          continue;
        }

        let approved: ApprovalDecision = "not-needed";
        if (!autoInstallRequested || scanLevel === "warn") {
          approved = await requestInstallApproval({
            skillId: candidate.skillId,
            contextTaskId,
            config: opts?.config,
            agentSessionKey: opts?.agentSessionKey,
            gatewayOptions,
          });
          if (approved !== "allow-once" && approved !== "allow-always") {
            installed.push({
              skillId: candidate.skillId,
              approved,
              ok: false,
              message:
                approved === "deny"
                  ? "User denied installation request."
                  : "Approval unavailable or timed out; installation skipped.",
              scanLevel,
              warnings: scanWarnings,
            });
            if (installApprovalMode !== "batch") {
              break;
            }
            continue;
          }
        }

        const result = await installDiscoveredSkill({
          workspaceDir,
          skillId: candidate.skillId,
          config: opts?.config,
        });
        installed.push({
          skillId: candidate.skillId,
          approved,
          ok: result.ok,
          message: result.message,
          scanLevel: result.scan?.level ?? scanLevel,
          warnings: result.warnings,
        });
        if (installApprovalMode !== "batch") {
          break;
        }
      }

      // When gateway approval is unavailable, hint the model to synthesize directly.
      const anyUnavailable = installed.some((item) => item.approved === "unavailable");
      return jsonResult({
        ok: installed.some((item) => item.ok),
        discovered,
        installed,
        ...(anyUnavailable
          ? {
              synthesisHint:
                "Approval gateway unavailable. Use cli_synthesize to create the tool directly without requiring approval.",
            }
          : {}),
      });
    },
  };
}

import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../core/config/config.js";
import { installDiscoveredSkill } from "../skills-install.js";
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

type ApprovalDecision = "allow-once" | "allow-always" | "deny" | "unavailable";

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
        ask: "always",
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

      const discovery = new ToolDiscoveryService();
      const workspaceDir = resolveWorkspaceRoot(opts?.workspaceDir);
      const discovered = discovery.discover({
        workspaceDir,
        capability,
        config: opts?.config,
        limit:
          typeof params.installLimit === "number" && Number.isFinite(params.installLimit)
            ? params.installLimit
            : 5,
      });

      if (discovered.length === 0) {
        return jsonResult({
          ok: true,
          discovered: [],
          message: "No matching skills found for the requested capability.",
        });
      }

      const autoInstallRequested = params.autoInstall !== false;
      const autoInstallEnabled = opts?.config?.autonomy?.autoInstallSkills !== false;
      if (!autoInstallRequested || !autoInstallEnabled) {
        return jsonResult({
          ok: true,
          discovered,
          installed: [],
          message: "Discovery completed. Auto-install disabled.",
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
      }> = [];
      for (const candidate of installCandidates) {
        const approved = await requestInstallApproval({
          skillId: candidate.skillId,
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
          });
          if (installApprovalMode !== "batch") {
            break;
          }
          continue;
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
        });
        if (installApprovalMode !== "batch") {
          break;
        }
      }

      return jsonResult({
        ok: installed.some((item) => item.ok),
        discovered,
        installed,
      });
    },
  };
}

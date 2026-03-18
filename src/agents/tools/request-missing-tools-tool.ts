import { execSync } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../core/config/config.js";
import { resolveAgentIdFromSessionKey } from "../../routing/session-key.js";
import { markInstalledSkillUsageRecord, readSkillUsageRecords } from "../skill-usage-records.js";
import {
  inspectDiscoveredSkillSafety,
  installDiscoveredSkill,
  type SkillInstallSafetyLevel,
} from "../skills-install.js";
import { resolveWorkspaceRoot } from "../workspace-dir.js";
import type { AnyAgentTool } from "./common.js";
import { jsonResult, readStringArrayParam, readStringParam } from "./common.js";
import { callGatewayTool, readGatewayCallOptions } from "./gateway.js";
import {
  ToolDiscoveryService,
  type DiscoveredSkill,
  type MissingCapability,
} from "./tool-discovery.js";

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

const EXTENSIONS_DIR = path.join(
  process.env.HOME ?? process.env.USERPROFILE ?? ".",
  ".marv",
  "extensions",
);

/** Install a skill discovered from a registry source (GitHub repo or npm package). */
async function installFromRegistrySource(
  candidate: DiscoveredSkill,
): Promise<{ ok: boolean; message: string }> {
  const install = candidate.registryInstall;
  if (!install) {
    return { ok: false, message: "No install spec available for this registry skill." };
  }
  const skillId = candidate.skillId;
  const installDir = path.join(EXTENSIONS_DIR, skillId);

  try {
    if (install.repo) {
      // Install from GitHub.
      await fs.mkdir(installDir, { recursive: true });
      execSync(`git clone --depth 1 ${install.repo} "${installDir}"`, {
        stdio: "pipe",
        timeout: 60_000,
      });
      // Run npm install if package.json exists.
      const pkgPath = path.join(installDir, "package.json");
      try {
        await fs.access(pkgPath);
        execSync("npm install --omit=dev", {
          cwd: installDir,
          stdio: "pipe",
          timeout: 120_000,
        });
      } catch {
        // No package.json or install failed — skill may not need deps.
      }
    } else if (install.npm) {
      // Install from npm.
      await fs.mkdir(installDir, { recursive: true });
      const tmpPkg = {
        name: `marv-skill-${skillId}`,
        version: "1.0.0",
        dependencies: { [install.npm]: "latest" },
      };
      await fs.writeFile(path.join(installDir, "package.json"), JSON.stringify(tmpPkg, null, 2));
      execSync("npm install --omit=dev", {
        cwd: installDir,
        stdio: "pipe",
        timeout: 120_000,
      });
    } else {
      return { ok: false, message: "Registry entry has no repo or npm install spec." };
    }

    await markInstalledSkillUsageRecord({ skillId });
    return {
      ok: true,
      message: `Skill '${skillId}' installed from registry. Restart gateway to activate plugin features.`,
    };
  } catch (err) {
    await fs.rm(installDir, { recursive: true, force: true }).catch(() => {});
    return {
      ok: false,
      message: `Failed to install '${skillId}': ${err instanceof Error ? err.message : String(err)}`,
    };
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
      const limit =
        typeof params.installLimit === "number" && Number.isFinite(params.installLimit)
          ? params.installLimit
          : 5;
      const discovered = await discovery.discoverAsync({
        workspaceDir,
        capability,
        config: opts?.config,
        usageRecords,
        limit,
        searchRegistries: true,
      });

      if (discovered.length === 0) {
        const synthesisEnabled = opts?.config?.autonomy?.toolSynthesis?.enabled !== false;
        return jsonResult({
          ok: true,
          discovered: [],
          synthesisHint: synthesisEnabled
            ? {
                guidance:
                  "No existing skill matches. Create one:\n" +
                  "1. Write a wrapper script (Python/Bash preferred).\n" +
                  "2. Test it via `exec`.\n" +
                  "3. Register with `cli_synthesize` — it auto-verifies and creates a discoverable skill entry.\n" +
                  "4. Future invocations via `cli_invoke`.",
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
        // Workspace/project skills and CLI profiles are already locally present.
        if (candidate.alreadyInstalled) {
          const hint =
            candidate.source === "cli-profile"
              ? `CLI profile already registered. Invoke via cli_invoke with profileId="${candidate.skillId}".`
              : "Skill already available (workspace source; no install required).";
          installed.push({
            skillId: candidate.skillId,
            approved: "not-needed",
            ok: true,
            message: hint,
          });
          if (installApprovalMode !== "batch") {
            break;
          }
          continue;
        }

        // Registry-sourced skills: install from GitHub/npm.
        if (candidate.source === "registry" && candidate.registryInstall) {
          let approved: ApprovalDecision = "not-needed";
          if (!autoInstallRequested) {
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
              });
              if (installApprovalMode !== "batch") {
                break;
              }
              continue;
            }
          }
          const result = await installFromRegistrySource(candidate);
          installed.push({
            skillId: candidate.skillId,
            approved,
            ok: result.ok,
            message: result.message,
          });
          if (installApprovalMode !== "batch") {
            break;
          }
          continue;
        }

        // Standard local skill installation with safety scanning.
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

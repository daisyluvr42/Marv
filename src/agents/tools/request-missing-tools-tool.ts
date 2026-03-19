import { execFileSync } from "node:child_process";
import type { Dirent } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../core/config/config.js";
import { resolveAgentIdFromSessionKey } from "../../routing/session-key.js";
import { scanDirectoryWithSummary, type SkillScanSummary } from "../../security/skill-scanner.js";
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
  collectPackageJsonFiles,
  detectLifecycleScripts,
  scanRegistryInstallDir,
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

// Only allow alphanumeric, hyphens, underscores, and dots in skill IDs.
const SAFE_SKILL_ID_RE = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;

// Validates that a repo URL looks like a legitimate git URL (HTTPS or SSH).
const SAFE_REPO_URL_RE = /^https:\/\/[a-zA-Z0-9._-]+\/[a-zA-Z0-9._/-]+(?:\.git)?$/;

// Validates that an npm package name follows the spec.
const SAFE_NPM_NAME_RE = /^(?:@[a-z0-9._-]+\/)?[a-z0-9._-]+$/;

function sanitizeSkillId(raw: string): string | null {
  const trimmed = raw.trim();
  return SAFE_SKILL_ID_RE.test(trimmed) ? trimmed : null;
}

function validateRepoUrl(raw: string): string | null {
  const trimmed = raw.trim();
  return SAFE_REPO_URL_RE.test(trimmed) ? trimmed : null;
}

function validateNpmName(raw: string): string | null {
  const trimmed = raw.trim();
  return SAFE_NPM_NAME_RE.test(trimmed) ? trimmed : null;
}

/**
 * Download and install a registry skill's dependencies WITHOUT executing
 * lifecycle scripts (postinstall, etc.).  The caller is responsible for
 * running a safety scan before calling `runRegistrySkillScripts()`.
 */
async function installFromRegistrySource(
  candidate: DiscoveredSkill,
): Promise<{ ok: boolean; message: string; installDir?: string }> {
  const install = candidate.registryInstall;
  if (!install) {
    return { ok: false, message: "No install spec available for this registry skill." };
  }

  const skillId = sanitizeSkillId(candidate.skillId);
  if (!skillId) {
    return {
      ok: false,
      message: `Skill ID '${candidate.skillId}' contains invalid characters and was rejected.`,
    };
  }
  const installDir = path.join(EXTENSIONS_DIR, skillId);

  try {
    if (install.repo) {
      const repoUrl = validateRepoUrl(install.repo);
      if (!repoUrl) {
        return {
          ok: false,
          message: `Registry repo URL '${install.repo}' failed validation (only HTTPS URLs allowed).`,
        };
      }
      // Install from GitHub using execFileSync to avoid shell injection.
      await fs.mkdir(installDir, { recursive: true });
      execFileSync("git", ["clone", "--depth", "1", repoUrl, installDir], {
        stdio: "pipe",
        timeout: 60_000,
      });
      // Install deps without executing lifecycle scripts (postinstall, etc.).
      // Scripts run only after the safety scan passes (see runRegistrySkillScripts).
      const pkgPath = path.join(installDir, "package.json");
      try {
        await fs.access(pkgPath);
        execFileSync("npm", ["install", "--omit=dev", "--ignore-scripts"], {
          cwd: installDir,
          stdio: "pipe",
          timeout: 120_000,
        });
      } catch {
        // No package.json or install failed — skill may not need deps.
      }
    } else if (install.npm) {
      const npmName = validateNpmName(install.npm);
      if (!npmName) {
        return {
          ok: false,
          message: `Registry npm package name '${install.npm}' failed validation.`,
        };
      }
      // Install from npm without executing lifecycle scripts.
      await fs.mkdir(installDir, { recursive: true });
      const tmpPkg = {
        name: `marv-skill-${skillId}`,
        version: "1.0.0",
        dependencies: { [npmName]: "latest" },
      };
      await fs.writeFile(path.join(installDir, "package.json"), JSON.stringify(tmpPkg, null, 2));
      execFileSync("npm", ["install", "--omit=dev", "--ignore-scripts"], {
        cwd: installDir,
        stdio: "pipe",
        timeout: 120_000,
      });
    } else {
      return { ok: false, message: "Registry entry has no repo or npm install spec." };
    }

    return {
      ok: true,
      installDir,
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

// npm lifecycle script keys that execute arbitrary code during install/rebuild.
const LIFECYCLE_SCRIPT_KEYS = new Set([
  "preinstall",
  "install",
  "postinstall",
  "preuninstall",
  "postuninstall",
  "prepublish",
  "prepare",
]);

/**
 * Detect lifecycle scripts in any package.json under the install directory
 * (root + nested node_modules). Returns the list of packages that have them.
 */
async function detectLifecycleScripts(
  installDir: string,
): Promise<{ pkg: string; scripts: string[] }[]> {
  const results: { pkg: string; scripts: string[] }[] = [];
  const packageJsonPaths = await collectPackageJsonFiles(installDir);
  for (const pkgPath of packageJsonPaths) {
    try {
      const raw = await fs.readFile(pkgPath, "utf-8");
      const parsed = JSON.parse(raw) as { name?: string; scripts?: Record<string, string> };
      if (!parsed.scripts) {
        continue;
      }
      const found = Object.keys(parsed.scripts).filter((k) => LIFECYCLE_SCRIPT_KEYS.has(k));
      if (found.length > 0) {
        results.push({ pkg: parsed.name ?? pkgPath, scripts: found });
      }
    } catch {
      // Missing or unreadable — skip.
    }
  }

  return results;
}

async function collectPackageJsonFiles(rootDir: string): Promise<string[]> {
  const out: string[] = [];
  const stack = [rootDir];

  while (stack.length > 0) {
    const currentDir = stack.pop();
    if (!currentDir) {
      continue;
    }

    let entries: Dirent[] = [];
    try {
      entries = await fs.readdir(currentDir, { encoding: "utf8", withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      if (entry.name.startsWith(".")) {
        continue;
      }
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }
      if (entry.isFile() && entry.name === "package.json") {
        out.push(fullPath);
      }
    }
  }

  return out;
}

type RegistryScanResult = {
  blocked: boolean;
  warnings: string[];
  scan: SkillScanSummary | null;
  lifecycleScripts: { pkg: string; scripts: string[] }[];
};

/**
 * Scan a registry skill's install directory directly (not via workspace
 * skill lookup). Also checks package.json files for lifecycle scripts.
 */
async function scanRegistryInstallDir(installDir: string): Promise<RegistryScanResult> {
  const warnings: string[] = [];
  let scan: SkillScanSummary | null = null;
  let blocked = false;
  const packageJsonFiles = await collectPackageJsonFiles(installDir);
  const includeFiles = packageJsonFiles.map((filePath) => path.relative(installDir, filePath));

  // 1. Scan source code for dangerous patterns.
  try {
    scan = await scanDirectoryWithSummary(installDir, {
      // Explicitly include every package.json so manifest contents are scanned
      // even though the default directory walk skips node_modules and non-code files.
      includeFiles,
      maxFiles: Math.max(500, includeFiles.length + 200),
    });
    if (scan.critical > 0) {
      blocked = true;
      for (const finding of scan.findings) {
        if (finding.severity === "critical") {
          warnings.push(`[${finding.ruleId}] ${finding.file}:${finding.line} — ${finding.message}`);
        }
      }
    }
  } catch {
    warnings.push("Failed to scan install directory.");
    blocked = true;
  }

  // 2. Detect lifecycle scripts — these would execute during npm rebuild.
  const lifecycleScripts = await detectLifecycleScripts(installDir);
  if (lifecycleScripts.length > 0) {
    blocked = true;
    for (const entry of lifecycleScripts) {
      warnings.push(
        `Package '${entry.pkg}' has lifecycle scripts (${entry.scripts.join(", ")}) — blocked to prevent arbitrary code execution during rebuild.`,
      );
    }
  }

  return { blocked, warnings, scan, lifecycleScripts };
}

/**
 * Run deferred lifecycle scripts (npm rebuild) for a registry skill that
 * has passed the safety scan AND has no lifecycle scripts.
 */
function runRegistrySkillScripts(installDir: string): void {
  try {
    execFileSync("npm", ["rebuild"], {
      cwd: installDir,
      stdio: "pipe",
      timeout: 120_000,
    });
  } catch {
    // rebuild failure is non-fatal — skill may not need native compilation.
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
        // Always require explicit approval for remote code installation.
        if (candidate.source === "registry" && candidate.registryInstall) {
          const approved = await requestInstallApproval({
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
          const result = await installFromRegistrySource(candidate);
          if (!result.ok) {
            installed.push({
              skillId: candidate.skillId,
              approved,
              ok: false,
              message: result.message,
            });
            if (installApprovalMode !== "batch") {
              break;
            }
            continue;
          }
          // Scan the actual install directory for dangerous patterns and
          // lifecycle scripts (not via workspace skill lookup — that can't
          // find registry skills under ~/.marv/extensions/).
          const registryScan = result.installDir
            ? await scanRegistryInstallDir(result.installDir)
            : null;
          if (registryScan?.blocked) {
            // Remove the installed skill — it didn't pass the scan.
            if (result.installDir) {
              await fs.rm(result.installDir, { recursive: true, force: true }).catch(() => {});
            }
            installed.push({
              skillId: candidate.skillId,
              approved,
              ok: false,
              message:
                registryScan.warnings[0] ?? "Registry skill blocked by safety scan after install.",
              warnings: registryScan.warnings,
            });
            if (installApprovalMode !== "batch") {
              break;
            }
            continue;
          }
          // Scan passed and no lifecycle scripts — safe to run npm rebuild
          // for native addons (node-gyp etc.).
          if (result.installDir && registryScan?.lifecycleScripts.length === 0) {
            runRegistrySkillScripts(result.installDir);
          }
          await markInstalledSkillUsageRecord({ skillId: candidate.skillId });
          installed.push({
            skillId: candidate.skillId,
            approved,
            ok: true,
            message: result.message,
            warnings: registryScan?.warnings,
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

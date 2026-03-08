import fs from "node:fs";
import path from "node:path";
import type { MarvConfig } from "../core/config/config.js";
import { resolveBrewExecutable } from "../infra/brew.js";
import { runCommandWithTimeout, type CommandOptions } from "../process/exec.js";
import { scanDirectoryWithSummary, type SkillScanFinding } from "../security/skill-scanner.js";
import { resolveUserPath } from "../utils.js";
import { markInstalledSkillUsageRecord } from "./skill-usage-records.js";
import { installDownloadSpec } from "./skills-install-download.js";
import { formatInstallFailureMessage } from "./skills-install-output.js";
import {
  hasBinary,
  loadWorkspaceSkillEntries,
  resolveSkillsInstallPreferences,
  type SkillEntry,
  type SkillInstallSpec,
  type SkillsInstallPreferences,
} from "./skills.js";

export type SkillInstallRequest = {
  workspaceDir: string;
  skillName: string;
  installId: string;
  timeoutMs?: number;
  config?: MarvConfig;
};

export type SkillInstallResult = {
  ok: boolean;
  message: string;
  stdout: string;
  stderr: string;
  code: number | null;
  warnings?: string[];
  scan?: SkillInstallSafetyReport;
};

export type DiscoverySkillInstallRequest = {
  workspaceDir: string;
  skillId: string;
  installId?: string;
  timeoutMs?: number;
  config?: MarvConfig;
};

export type SkillInstallSafetyLevel = "clean" | "warn" | "critical";

export type SkillInstallSafetyReport = {
  level: SkillInstallSafetyLevel;
  warnings: string[];
  findings: SkillScanFinding[];
  blocked: boolean;
};

function withWarnings(result: SkillInstallResult, warnings: string[]): SkillInstallResult {
  if (warnings.length === 0) {
    return result;
  }
  return {
    ...result,
    warnings: warnings.slice(),
  };
}

function formatScanFindingDetail(
  rootDir: string,
  finding: { message: string; file: string; line: number },
): string {
  const relativePath = path.relative(rootDir, finding.file);
  const filePath =
    relativePath && relativePath !== "." && !relativePath.startsWith("..")
      ? relativePath
      : path.basename(finding.file);
  return `${finding.message} (${filePath}:${finding.line})`;
}

function withScan(
  result: SkillInstallResult,
  scan: SkillInstallSafetyReport | undefined,
): SkillInstallResult {
  if (!scan) {
    return result;
  }
  return {
    ...result,
    scan,
  };
}

function buildCriticalWarnings(
  skillName: string,
  skillDir: string,
  findings: SkillScanFinding[],
): string[] {
  const criticalDetails = findings
    .filter((finding) => finding.severity === "critical")
    .map((finding) => formatScanFindingDetail(skillDir, finding))
    .join("; ");
  return [`WARNING: Skill "${skillName}" contains dangerous code patterns: ${criticalDetails}`];
}

function buildWarnWarnings(skillName: string, warnCount: number): string[] {
  return [
    `Skill "${skillName}" has ${warnCount} suspicious code pattern(s). Run "marv security audit --deep" for details.`,
  ];
}

export async function inspectSkillInstallSafety(
  entry: SkillEntry,
): Promise<SkillInstallSafetyReport> {
  const skillName = entry.skill.name;
  const skillDir = path.resolve(entry.skill.baseDir);

  try {
    const summary = await scanDirectoryWithSummary(skillDir);
    if (summary.critical > 0) {
      return {
        level: "critical",
        warnings: buildCriticalWarnings(skillName, skillDir, summary.findings),
        findings: summary.findings,
        blocked: true,
      };
    }
    if (summary.warn > 0) {
      return {
        level: "warn",
        warnings: buildWarnWarnings(skillName, summary.warn),
        findings: summary.findings,
        blocked: false,
      };
    }
    return {
      level: "clean",
      warnings: [],
      findings: summary.findings,
      blocked: false,
    };
  } catch (err) {
    return {
      level: "warn",
      warnings: [
        `Skill "${skillName}" code safety scan failed (${String(err)}). Review before install or run "marv security audit --deep".`,
      ],
      findings: [],
      blocked: false,
    };
  }
}

function resolveInstallId(spec: SkillInstallSpec, index: number): string {
  return (spec.id ?? `${spec.kind}-${index}`).trim();
}

function findInstallSpec(entry: SkillEntry, installId: string): SkillInstallSpec | undefined {
  const specs = entry.metadata?.install ?? [];
  for (const [index, spec] of specs.entries()) {
    if (resolveInstallId(spec, index) === installId) {
      return spec;
    }
  }
  return undefined;
}

function buildNodeInstallCommand(packageName: string, prefs: SkillsInstallPreferences): string[] {
  switch (prefs.nodeManager) {
    case "pnpm":
      return ["pnpm", "add", "-g", "--ignore-scripts", packageName];
    case "yarn":
      return ["yarn", "global", "add", "--ignore-scripts", packageName];
    case "bun":
      return ["bun", "add", "-g", "--ignore-scripts", packageName];
    default:
      return ["npm", "install", "-g", "--ignore-scripts", packageName];
  }
}

function buildInstallCommand(
  spec: SkillInstallSpec,
  prefs: SkillsInstallPreferences,
): {
  argv: string[] | null;
  error?: string;
} {
  switch (spec.kind) {
    case "brew": {
      if (!spec.formula) {
        return { argv: null, error: "missing brew formula" };
      }
      return { argv: ["brew", "install", spec.formula] };
    }
    case "node": {
      if (!spec.package) {
        return { argv: null, error: "missing node package" };
      }
      return {
        argv: buildNodeInstallCommand(spec.package, prefs),
      };
    }
    case "go": {
      if (!spec.module) {
        return { argv: null, error: "missing go module" };
      }
      return { argv: ["go", "install", spec.module] };
    }
    case "uv": {
      if (!spec.package) {
        return { argv: null, error: "missing uv package" };
      }
      return { argv: ["uv", "tool", "install", spec.package] };
    }
    case "download": {
      return { argv: null, error: "download install handled separately" };
    }
    default:
      return { argv: null, error: "unsupported installer" };
  }
}

async function resolveBrewBinDir(timeoutMs: number, brewExe?: string): Promise<string | undefined> {
  const exe = brewExe ?? (hasBinary("brew") ? "brew" : resolveBrewExecutable());
  if (!exe) {
    return undefined;
  }

  const prefixResult = await runCommandWithTimeout([exe, "--prefix"], {
    timeoutMs: Math.min(timeoutMs, 30_000),
  });
  if (prefixResult.code === 0) {
    const prefix = prefixResult.stdout.trim();
    if (prefix) {
      return path.join(prefix, "bin");
    }
  }

  const envPrefix = process.env.HOMEBREW_PREFIX?.trim();
  if (envPrefix) {
    return path.join(envPrefix, "bin");
  }

  for (const candidate of ["/opt/homebrew/bin", "/usr/local/bin"]) {
    try {
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    } catch {
      // ignore
    }
  }
  return undefined;
}

type CommandResult = {
  code: number | null;
  stdout: string;
  stderr: string;
};

function createInstallFailure(params: {
  message: string;
  stdout?: string;
  stderr?: string;
  code?: number | null;
}): SkillInstallResult {
  return {
    ok: false,
    message: params.message,
    stdout: params.stdout?.trim() ?? "",
    stderr: params.stderr?.trim() ?? "",
    code: params.code ?? null,
  };
}

function createInstallSuccess(result: CommandResult): SkillInstallResult {
  return {
    ok: true,
    message: "Installed",
    stdout: result.stdout.trim(),
    stderr: result.stderr.trim(),
    code: result.code,
  };
}

async function runCommandSafely(
  argv: string[],
  optionsOrTimeout: number | CommandOptions,
): Promise<CommandResult> {
  try {
    const result = await runCommandWithTimeout(argv, optionsOrTimeout);
    return {
      code: result.code,
      stdout: result.stdout,
      stderr: result.stderr,
    };
  } catch (err) {
    return {
      code: null,
      stdout: "",
      stderr: err instanceof Error ? err.message : String(err),
    };
  }
}

async function runBestEffortCommand(
  argv: string[],
  optionsOrTimeout: number | CommandOptions,
): Promise<void> {
  await runCommandSafely(argv, optionsOrTimeout);
}

function resolveBrewMissingFailure(spec: SkillInstallSpec): SkillInstallResult {
  const formula = spec.formula ?? "this package";
  const hint =
    process.platform === "linux"
      ? `Homebrew is not installed. Install it from https://brew.sh or install "${formula}" manually using your system package manager (e.g. apt, dnf, pacman).`
      : "Homebrew is not installed. Install it from https://brew.sh";
  return createInstallFailure({ message: `brew not installed — ${hint}` });
}

async function ensureUvInstalled(params: {
  spec: SkillInstallSpec;
  brewExe?: string;
  timeoutMs: number;
}): Promise<SkillInstallResult | undefined> {
  if (params.spec.kind !== "uv" || hasBinary("uv")) {
    return undefined;
  }

  if (!params.brewExe) {
    return createInstallFailure({
      message:
        "uv not installed — install manually: https://docs.astral.sh/uv/getting-started/installation/",
    });
  }

  const brewResult = await runCommandSafely([params.brewExe, "install", "uv"], {
    timeoutMs: params.timeoutMs,
  });
  if (brewResult.code === 0) {
    return undefined;
  }

  return createInstallFailure({
    message: "Failed to install uv (brew)",
    ...brewResult,
  });
}

async function installGoViaApt(timeoutMs: number): Promise<SkillInstallResult | undefined> {
  const aptInstallArgv = ["apt-get", "install", "-y", "golang-go"];
  const aptUpdateArgv = ["apt-get", "update", "-qq"];
  const aptFailureMessage =
    "go not installed — automatic install via apt failed. Install manually: https://go.dev/doc/install";

  const isRoot = typeof process.getuid === "function" && process.getuid() === 0;
  if (isRoot) {
    // Best effort: fresh containers often need package indexes populated.
    await runBestEffortCommand(aptUpdateArgv, { timeoutMs });
    const aptResult = await runCommandSafely(aptInstallArgv, { timeoutMs });
    if (aptResult.code === 0) {
      return undefined;
    }
    return createInstallFailure({
      message: aptFailureMessage,
      ...aptResult,
    });
  }

  if (!hasBinary("sudo")) {
    return createInstallFailure({
      message:
        "go not installed — apt-get is available but sudo is not installed. Install manually: https://go.dev/doc/install",
    });
  }

  const sudoCheck = await runCommandSafely(["sudo", "-n", "true"], {
    timeoutMs: 5_000,
  });
  if (sudoCheck.code !== 0) {
    return createInstallFailure({
      message:
        "go not installed — apt-get is available but sudo is not usable (missing or requires a password). Install manually: https://go.dev/doc/install",
      ...sudoCheck,
    });
  }

  // Best effort: fresh containers often need package indexes populated.
  await runBestEffortCommand(["sudo", ...aptUpdateArgv], { timeoutMs });
  const aptResult = await runCommandSafely(["sudo", ...aptInstallArgv], {
    timeoutMs,
  });
  if (aptResult.code === 0) {
    return undefined;
  }

  return createInstallFailure({
    message: aptFailureMessage,
    ...aptResult,
  });
}

async function ensureGoInstalled(params: {
  spec: SkillInstallSpec;
  brewExe?: string;
  timeoutMs: number;
}): Promise<SkillInstallResult | undefined> {
  if (params.spec.kind !== "go" || hasBinary("go")) {
    return undefined;
  }

  if (params.brewExe) {
    const brewResult = await runCommandSafely([params.brewExe, "install", "go"], {
      timeoutMs: params.timeoutMs,
    });
    if (brewResult.code === 0) {
      return undefined;
    }
    return createInstallFailure({
      message: "Failed to install go (brew)",
      ...brewResult,
    });
  }

  if (hasBinary("apt-get")) {
    return installGoViaApt(params.timeoutMs);
  }

  return createInstallFailure({
    message: "go not installed — install manually: https://go.dev/doc/install",
  });
}

async function executeInstallCommand(params: {
  argv: string[] | null;
  timeoutMs: number;
  env?: NodeJS.ProcessEnv;
}): Promise<SkillInstallResult> {
  if (!params.argv || params.argv.length === 0) {
    return createInstallFailure({ message: "invalid install command" });
  }

  const result = await runCommandSafely(params.argv, {
    timeoutMs: params.timeoutMs,
    env: params.env,
  });
  if (result.code === 0) {
    return createInstallSuccess(result);
  }

  return createInstallFailure({
    message: formatInstallFailureMessage(result),
    ...result,
  });
}

export async function installSkill(params: SkillInstallRequest): Promise<SkillInstallResult> {
  const timeoutMs = Math.min(Math.max(params.timeoutMs ?? 300_000, 1_000), 900_000);
  const workspaceDir = resolveUserPath(params.workspaceDir);
  const entries = loadWorkspaceSkillEntries(workspaceDir);
  const entry = entries.find((item) => item.skill.name === params.skillName);
  if (!entry) {
    return {
      ok: false,
      message: `Skill not found: ${params.skillName}`,
      stdout: "",
      stderr: "",
      code: null,
    };
  }

  const spec = findInstallSpec(entry, params.installId);
  const scan = await inspectSkillInstallSafety(entry);
  const warnings = scan.warnings;
  if (scan.blocked) {
    return withScan(
      withWarnings(
        {
          ok: false,
          message: `Installation blocked: skill "${entry.skill.name}" failed the safety scan.`,
          stdout: "",
          stderr: "",
          code: null,
        },
        warnings,
      ),
      scan,
    );
  }
  if (!spec) {
    return withScan(
      withWarnings(
        {
          ok: false,
          message: `Installer not found: ${params.installId}`,
          stdout: "",
          stderr: "",
          code: null,
        },
        warnings,
      ),
      scan,
    );
  }
  if (spec.kind === "download") {
    const downloadResult = await installDownloadSpec({ entry, spec, timeoutMs });
    return withScan(withWarnings(downloadResult, warnings), scan);
  }

  const prefs = resolveSkillsInstallPreferences(params.config);
  const command = buildInstallCommand(spec, prefs);
  if (command.error) {
    return withScan(
      withWarnings(
        {
          ok: false,
          message: command.error,
          stdout: "",
          stderr: "",
          code: null,
        },
        warnings,
      ),
      scan,
    );
  }

  const brewExe = hasBinary("brew") ? "brew" : resolveBrewExecutable();
  if (spec.kind === "brew" && !brewExe) {
    return withScan(withWarnings(resolveBrewMissingFailure(spec), warnings), scan);
  }

  const uvInstallFailure = await ensureUvInstalled({ spec, brewExe, timeoutMs });
  if (uvInstallFailure) {
    return withScan(withWarnings(uvInstallFailure, warnings), scan);
  }

  const goInstallFailure = await ensureGoInstalled({ spec, brewExe, timeoutMs });
  if (goInstallFailure) {
    return withScan(withWarnings(goInstallFailure, warnings), scan);
  }

  const argv = command.argv ? [...command.argv] : null;
  if (spec.kind === "brew" && brewExe && argv?.[0] === "brew") {
    argv[0] = brewExe;
  }

  let env: NodeJS.ProcessEnv | undefined;
  if (spec.kind === "go" && brewExe) {
    const brewBin = await resolveBrewBinDir(timeoutMs, brewExe);
    if (brewBin) {
      env = { GOBIN: brewBin };
    }
  }

  const result = withScan(
    withWarnings(await executeInstallCommand({ argv, timeoutMs, env }), warnings),
    scan,
  );
  if (result.ok) {
    await markInstalledSkillUsageRecord({
      skillId: entry.skill.name,
    }).catch(() => undefined);
  }
  return result;
}

export async function installDiscoveredSkill(
  params: DiscoverySkillInstallRequest,
): Promise<SkillInstallResult> {
  const workspaceDir = resolveUserPath(params.workspaceDir);
  const entries = loadWorkspaceSkillEntries(workspaceDir, { config: params.config });
  const entry = entries.find((item) => item.skill.name === params.skillId);
  if (!entry) {
    return {
      ok: false,
      message: `Discovered skill not found: ${params.skillId}`,
      stdout: "",
      stderr: "",
      code: null,
    };
  }

  const installSpecs = entry.metadata?.install ?? [];
  if (installSpecs.length === 0) {
    return {
      ok: false,
      message: `Discovered skill has no install steps: ${params.skillId}`,
      stdout: "",
      stderr: "",
      code: null,
    };
  }

  const selectedInstallId = params.installId?.trim() || resolveInstallId(installSpecs[0], 0);

  return installSkill({
    workspaceDir,
    skillName: entry.skill.name,
    installId: selectedInstallId,
    timeoutMs: params.timeoutMs,
    config: params.config,
  });
}

export async function inspectDiscoveredSkillSafety(
  params: DiscoverySkillInstallRequest,
): Promise<SkillInstallSafetyReport | null> {
  const workspaceDir = resolveUserPath(params.workspaceDir);
  const entries = loadWorkspaceSkillEntries(workspaceDir, { config: params.config });
  const entry = entries.find((item) => item.skill.name === params.skillId);
  if (!entry) {
    return null;
  }
  return await inspectSkillInstallSafety(entry);
}

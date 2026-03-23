import fs from "node:fs/promises";
import path from "node:path";
import { cancel, confirm, isCancel, multiselect, password } from "@clack/prompts";
import { withProgress } from "../cli/progress.js";
import { parseConfigJson5 } from "../core/config/config.js";
import type { RuntimeEnv } from "../runtime.js";
import { stylePromptHint, stylePromptMessage, stylePromptTitle } from "../terminal/prompt-style.js";
import { shortenHomePath } from "../utils.js";
import { resolveCleanupPlanFromDisk } from "./cleanup-plan.js";
import { buildCleanupPlan } from "./cleanup-utils.js";
import {
  extractMigrateArchive,
  isEncryptedMigrateArchive,
  readArchiveManifest,
} from "./migrate-archive.js";
import {
  MIGRATE_ARCHIVE_ROOT,
  parseMigrateScopesFlag,
  type MigrateImportOptions,
  type MigrateManifest,
  type MigrateManifestItem,
  type MigrateScope,
  SCOPE_OPTIONS,
} from "./migrate-types.js";

const multiselectStyled = <T>(params: Parameters<typeof multiselect<T>>[0]) =>
  multiselect({
    ...params,
    message: stylePromptMessage(params.message),
    options: params.options.map((opt) =>
      opt.hint === undefined ? opt : { ...opt, hint: stylePromptHint(opt.hint) },
    ),
  });

type ImportPlanEntry = {
  scope: MigrateScope;
  sourcePath: string;
  targetPath: string;
  kind: "file" | "directory";
};

export async function migrateImportCommand(runtime: RuntimeEnv, opts: MigrateImportOptions) {
  const interactive = !opts.nonInteractive;
  let archivePassword = opts.password;

  if (isEncryptedMigrateArchive(opts.archivePath) && !archivePassword) {
    if (!interactive) {
      runtime.error("Encrypted archives require --password in non-interactive mode.");
      runtime.exit(1);
      return;
    }
    const entered = await password({
      message: stylePromptMessage("Archive password"),
      validate: (value) => (String(value ?? "").trim() ? undefined : "Required"),
    });
    if (isCancel(entered)) {
      cancel(stylePromptTitle("Migration import cancelled.") ?? "Migration import cancelled.");
      runtime.exit(0);
      return;
    }
    archivePassword = String(entered);
  }

  const manifest = await readArchiveManifest(opts.archivePath, archivePassword);
  logManifestSummary(runtime, manifest);

  const availableScopes = manifest.scopes;
  let scopes = opts.scopes?.filter((scope) => availableScopes.includes(scope));
  if (!scopes?.length) {
    if (!interactive || opts.yes) {
      scopes = [...availableScopes];
    } else {
      const selection = await multiselectStyled<MigrateScope>({
        message: "Import which scopes?",
        options: SCOPE_OPTIONS.filter((option) => availableScopes.includes(option.value)),
        initialValues: [...availableScopes],
        required: true,
      });
      if (isCancel(selection)) {
        cancel(stylePromptTitle("Migration import cancelled.") ?? "Migration import cancelled.");
        runtime.exit(0);
        return;
      }
      scopes = selection;
    }
  }

  if (scopes.length === 0) {
    runtime.log("Nothing selected.");
    return;
  }

  const extractDir = await fs.mkdtemp(
    path.join(path.dirname(path.resolve(opts.archivePath)), ".marv-migrate-import-"),
  );
  try {
    await withProgress({ label: "Extracting migration archive" }, async () => {
      await extractMigrateArchive(opts.archivePath, extractDir, archivePassword);
    });

    const currentPlan = resolveCleanupPlanFromDisk();
    const effectiveWorkspaceDirs = await resolveImportWorkspaceDirs({
      extractDir,
      scopes,
      currentPlan,
    });
    let importEntries = buildImportPlan({
      manifest,
      extractDir,
      scopes,
      stateDir: currentPlan.stateDir,
      configPath: currentPlan.configPath,
      oauthDir: currentPlan.oauthDir,
      workspaceDirs: effectiveWorkspaceDirs,
    });
    const conflicts = await collectScopeConflicts(importEntries);

    if (opts.dryRun) {
      runtime.log(`[dry-run] ${importEntries.length} archive item(s) would be imported.`);
      if (conflicts.length === 0) {
        runtime.log("[dry-run] No conflicts detected.");
      } else {
        for (const conflict of conflicts) {
          runtime.log(
            `[dry-run] ${conflict.scope}: ${conflict.paths.map((entry) => shortenHomePath(entry)).join(", ")}`,
          );
        }
      }
      return;
    }

    if (conflicts.length > 0 && !opts.force) {
      if (!interactive) {
        runtime.error("Import would overwrite existing files. Re-run with --force.");
        runtime.exit(1);
        return;
      }
      const allowedScopes = new Set<MigrateScope>(scopes);
      for (const conflict of conflicts) {
        const approved = await confirm({
          message: stylePromptMessage(
            `Overwrite existing ${conflict.scope} data? (${conflict.paths.map((entry) => shortenHomePath(entry)).join(", ")})`,
          ),
        });
        if (isCancel(approved)) {
          cancel(stylePromptTitle("Migration import cancelled.") ?? "Migration import cancelled.");
          runtime.exit(0);
          return;
        }
        if (!approved) {
          allowedScopes.delete(conflict.scope);
        }
      }
      importEntries = importEntries.filter((entry) => allowedScopes.has(entry.scope));
    }

    if (importEntries.length === 0) {
      runtime.log("Nothing to import after conflict checks.");
      return;
    }

    await withProgress({ label: "Importing migration data" }, async (progress) => {
      for (const entry of importEntries) {
        progress.setLabel(`Importing ${entry.scope}`);
        await removeIfExists(entry.targetPath);
        await fs.mkdir(path.dirname(entry.targetPath), { recursive: true });
        if (entry.kind === "directory") {
          await fs.cp(entry.sourcePath, entry.targetPath, { recursive: true });
        } else {
          await fs.copyFile(entry.sourcePath, entry.targetPath);
        }
      }
    });

    runtime.log(
      `Imported scopes: ${[...new Set(importEntries.map((entry) => entry.scope))].join(", ")}`,
    );
  } finally {
    await fs.rm(extractDir, { recursive: true, force: true });
  }
}

export function parseMigrateImportOptions(
  archivePath: string,
  raw: {
    scopes?: string;
    password?: string;
    force?: boolean;
    dryRun?: boolean;
    yes?: boolean;
    nonInteractive?: boolean;
  },
): MigrateImportOptions {
  return {
    archivePath,
    scopes: parseMigrateScopesFlag(raw.scopes),
    password: raw.password?.trim() || undefined,
    force: Boolean(raw.force),
    dryRun: Boolean(raw.dryRun),
    yes: Boolean(raw.yes),
    nonInteractive: Boolean(raw.nonInteractive),
  };
}

function logManifestSummary(runtime: RuntimeEnv, manifest: MigrateManifest) {
  runtime.log(`Archive created: ${manifest.exportedAt}`);
  runtime.log(`Marv version: ${manifest.marvVersion}`);
  runtime.log(`Source host: ${manifest.hostname} (${manifest.platform})`);
  runtime.log(`Available scopes: ${manifest.scopes.join(", ")}`);
}

async function resolveImportWorkspaceDirs(params: {
  extractDir: string;
  scopes: readonly MigrateScope[];
  currentPlan: ReturnType<typeof resolveCleanupPlanFromDisk>;
}): Promise<string[]> {
  if (!params.scopes.includes("workspace") || !params.scopes.includes("config")) {
    return params.currentPlan.workspaceDirs;
  }
  const configDir = path.join(params.extractDir, MIGRATE_ARCHIVE_ROOT, "config");
  let entries: string[];
  try {
    entries = await fs.readdir(configDir);
  } catch {
    return params.currentPlan.workspaceDirs;
  }
  const configFile = entries[0];
  if (!configFile) {
    return params.currentPlan.workspaceDirs;
  }
  try {
    const raw = await fs.readFile(path.join(configDir, configFile), "utf-8");
    const parsed = parseConfigJson5(raw);
    if (!parsed.ok) {
      return params.currentPlan.workspaceDirs;
    }
    const derived = buildCleanupPlan({
      cfg: parsed.parsed as Parameters<typeof buildCleanupPlan>[0]["cfg"],
      stateDir: params.currentPlan.stateDir,
      configPath: params.currentPlan.configPath,
      oauthDir: params.currentPlan.oauthDir,
    });
    return derived.workspaceDirs;
  } catch {
    return params.currentPlan.workspaceDirs;
  }
}

function buildImportPlan(params: {
  manifest: MigrateManifest;
  extractDir: string;
  scopes: readonly MigrateScope[];
  stateDir: string;
  configPath: string;
  oauthDir: string;
  workspaceDirs: string[];
}): ImportPlanEntry[] {
  const wantedScopes = new Set(params.scopes);
  return params.manifest.items
    .filter((item) => wantedScopes.has(item.scope))
    .map((item) =>
      resolveImportPlanEntry({
        item,
        extractDir: params.extractDir,
        stateDir: params.stateDir,
        configPath: params.configPath,
        oauthDir: params.oauthDir,
        workspaceDirs: params.workspaceDirs,
      }),
    );
}

function resolveImportPlanEntry(params: {
  item: MigrateManifestItem;
  extractDir: string;
  stateDir: string;
  configPath: string;
  oauthDir: string;
  workspaceDirs: string[];
}): ImportPlanEntry {
  const sourcePath = path.join(params.extractDir, params.item.archivePath);
  const normalized = params.item.archivePath.replaceAll("\\", "/");
  const prefix = `${MIGRATE_ARCHIVE_ROOT}/${params.item.scope}`;
  if (!normalized.startsWith(prefix)) {
    throw new Error(`Unexpected archive path: ${params.item.archivePath}`);
  }

  if (params.item.scope === "config") {
    return {
      scope: "config",
      sourcePath,
      targetPath: params.configPath,
      kind: params.item.kind,
    };
  }

  if (params.item.scope === "workspace") {
    const segments = normalized.split("/");
    const slotRaw = segments[2];
    const slot = Number.parseInt(slotRaw ?? "", 10);
    if (!Number.isInteger(slot) || slot < 0) {
      throw new Error(`Workspace archive path is missing a slot: ${params.item.archivePath}`);
    }
    const workspaceDir = params.workspaceDirs[slot];
    if (!workspaceDir) {
      throw new Error(
        `No local workspace target is configured for exported workspace slot ${slot}.`,
      );
    }
    const relative = segments.slice(3).join("/");
    return {
      scope: "workspace",
      sourcePath,
      targetPath: relative ? path.join(workspaceDir, relative) : workspaceDir,
      kind: params.item.kind,
    };
  }

  const relative = normalized.slice(prefix.length).replace(/^\/+/, "");

  // Memory scope has sub-paths that map to different stateDir locations:
  // - "soul" → <stateDir>/memory/soul (SQLite DBs)
  // - "soul-files" → <stateDir>/soul (Soul.md identity files)
  // - "experience" → <stateDir>/experience (MARV_EXPERIENCE.md etc.)
  if (params.item.scope === "memory") {
    const targetPath = resolveMemoryImportTarget(relative, params.stateDir);
    return {
      scope: "memory",
      sourcePath,
      targetPath,
      kind: params.item.kind,
    };
  }

  const targetBase = resolveScopeTargetBase(params.item.scope, {
    stateDir: params.stateDir,
    oauthDir: params.oauthDir,
  });
  return {
    scope: params.item.scope,
    sourcePath,
    targetPath: relative ? path.join(targetBase, relative) : targetBase,
    kind: params.item.kind,
  };
}

function resolveScopeTargetBase(
  scope: Exclude<MigrateScope, "config" | "workspace" | "memory">,
  params: { stateDir: string; oauthDir: string },
): string {
  switch (scope) {
    case "sessions":
      return params.stateDir;
    case "credentials":
      return params.oauthDir;
    case "tasks":
      return path.join(params.stateDir, "tasks");
    case "ledger":
      return path.join(params.stateDir, "ledger");
  }
}

/**
 * Memory scope sub-paths map to different stateDir locations:
 * - "soul/..." → <stateDir>/memory/soul (SQLite DBs)
 * - "soul-files/..." → <stateDir>/soul (Soul.md identity files)
 * - "experience/..." → <stateDir>/experience (behavioral memory files)
 */
function resolveMemoryImportTarget(relative: string, stateDir: string): string {
  if (relative.startsWith("soul-files")) {
    const rest = relative.slice("soul-files".length).replace(/^\/+/, "");
    return rest ? path.join(stateDir, "soul", rest) : path.join(stateDir, "soul");
  }
  if (relative.startsWith("experience")) {
    const rest = relative.slice("experience".length).replace(/^\/+/, "");
    return rest ? path.join(stateDir, "experience", rest) : path.join(stateDir, "experience");
  }
  // Default: "soul" → <stateDir>/memory/soul (existing SQLite path)
  return relative ? path.join(stateDir, "memory", relative) : path.join(stateDir, "memory");
}

async function collectScopeConflicts(
  entries: readonly ImportPlanEntry[],
): Promise<Array<{ scope: MigrateScope; paths: string[] }>> {
  const grouped = new Map<MigrateScope, string[]>();
  for (const entry of entries) {
    if (!(await pathExists(entry.targetPath))) {
      continue;
    }
    const current = grouped.get(entry.scope) ?? [];
    current.push(entry.targetPath);
    grouped.set(entry.scope, current);
  }
  return [...grouped.entries()].map(([scope, paths]) => ({
    scope,
    paths,
  }));
}

async function removeIfExists(targetPath: string) {
  if (!(await pathExists(targetPath))) {
    return;
  }
  await fs.rm(targetPath, { recursive: true, force: true });
}

async function pathExists(targetPath: string): Promise<boolean> {
  try {
    await fs.lstat(targetPath);
    return true;
  } catch {
    return false;
  }
}

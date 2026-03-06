import fs from "node:fs/promises";
import path from "node:path";
import { cancel, confirm, isCancel, multiselect, password, text } from "@clack/prompts";
import { withProgress } from "../cli/progress.js";
import { resolveSessionTranscriptsDirForAgent } from "../core/config/sessions/paths.js";
import { resolveLedgerDbPath } from "../ledger/event-store.js";
import type { RuntimeEnv } from "../runtime.js";
import { selectStyled } from "../terminal/prompt-select-styled.js";
import { stylePromptHint, stylePromptMessage, stylePromptTitle } from "../terminal/prompt-style.js";
import { shortenHomePath } from "../utils.js";
import { VERSION } from "../version.js";
import { resolveCleanupPlanFromDisk } from "./cleanup-plan.js";
import { createMigrateArchive, encryptArchive } from "./migrate-archive.js";
import {
  createMigrateManifest,
  getDefaultMigrateExportPath,
  MIGRATE_DEFAULT_EXPORT_SCOPES,
  parseMigrateScopesFlag,
  resolveMigrateArchivePath,
  SCOPE_OPTIONS,
  type MigrateArchiveEntry,
  type MigrateExportOptions,
  type MigrateFormat,
  type MigrateScope,
} from "./migrate-types.js";

const multiselectStyled = <T>(params: Parameters<typeof multiselect<T>>[0]) =>
  multiselect({
    ...params,
    message: stylePromptMessage(params.message),
    options: params.options.map((opt) =>
      opt.hint === undefined ? opt : { ...opt, hint: stylePromptHint(opt.hint) },
    ),
  });

export async function migrateExportCommand(runtime: RuntimeEnv, opts: MigrateExportOptions) {
  const interactive = !opts.nonInteractive;
  const autoAccept = Boolean(opts.yes);

  if (!interactive && !opts.scopes?.length) {
    runtime.error("Non-interactive mode requires --scopes.");
    runtime.exit(1);
    return;
  }

  let scopes = opts.scopes;
  if (!scopes?.length) {
    if (!interactive || autoAccept) {
      scopes = [...MIGRATE_DEFAULT_EXPORT_SCOPES];
    } else {
      const selection = await multiselectStyled<MigrateScope>({
        message: "Export which data?",
        options: SCOPE_OPTIONS,
        initialValues: [...MIGRATE_DEFAULT_EXPORT_SCOPES],
        required: true,
      });
      if (isCancel(selection)) {
        cancel(stylePromptTitle("Migration export cancelled.") ?? "Migration export cancelled.");
        runtime.exit(0);
        return;
      }
      scopes = selection;
    }
  }

  if (scopes.includes("credentials") && interactive && !autoAccept) {
    const includeCredentials = await confirm({
      message: stylePromptMessage(
        "Credentials include tokens and API keys. Include them in this export?",
      ),
    });
    if (isCancel(includeCredentials) || !includeCredentials) {
      scopes = scopes.filter((scope) => scope !== "credentials");
    }
  }

  if (scopes.length === 0) {
    runtime.log("Nothing selected.");
    return;
  }

  let format = opts.format ?? "plain";
  if (!opts.format && interactive && !autoAccept) {
    const selection = await selectStyled<MigrateFormat>({
      message: "Archive format",
      options: [
        {
          value: "plain",
          label: "Plain tar.gz",
          hint: "portable but not encrypted",
        },
        {
          value: "encrypted",
          label: "Encrypted tar.gz.enc",
          hint: "AES-256-GCM with a password",
        },
      ],
      initialValue: "plain",
    });
    if (isCancel(selection)) {
      cancel(stylePromptTitle("Migration export cancelled.") ?? "Migration export cancelled.");
      runtime.exit(0);
      return;
    }
    format = selection;
  }

  let archivePassword = opts.password;
  if (format === "encrypted" && !archivePassword) {
    if (!interactive) {
      runtime.error("Encrypted exports require --password in non-interactive mode.");
      runtime.exit(1);
      return;
    }
    const entered = await password({
      message: stylePromptMessage("Archive password"),
      validate: (value) => (String(value ?? "").trim() ? undefined : "Required"),
    });
    if (isCancel(entered)) {
      cancel(stylePromptTitle("Migration export cancelled.") ?? "Migration export cancelled.");
      runtime.exit(0);
      return;
    }
    archivePassword = String(entered);
  }

  const defaultOutput = getDefaultMigrateExportPath(format);
  let outputPath = opts.output;
  if (!outputPath) {
    if (!interactive || autoAccept) {
      outputPath = defaultOutput;
    } else {
      const entered = await text({
        message: stylePromptMessage("Output archive path"),
        initialValue: defaultOutput,
        validate: (value) => (String(value ?? "").trim() ? undefined : "Required"),
      });
      if (isCancel(entered)) {
        cancel(stylePromptTitle("Migration export cancelled.") ?? "Migration export cancelled.");
        runtime.exit(0);
        return;
      }
      outputPath = String(entered).trim();
    }
  }

  const entries = await collectMigrateArchiveEntries(scopes);
  const manifest = createMigrateManifest({
    scopes,
    format,
    marvVersion: VERSION,
    items: entries.map(({ scope, archivePath, kind }) => ({ scope, archivePath, kind })),
  });

  await withProgress({ label: "Creating migration archive" }, async (progress) => {
    const resolvedOutput = path.resolve(outputPath);
    if (format === "plain") {
      await createMigrateArchive(entries, manifest, resolvedOutput);
      return;
    }
    const tempOutput = `${resolvedOutput}.tmp-${Date.now().toString(36)}`;
    try {
      progress.setLabel("Packing archive");
      await createMigrateArchive(entries, manifest, tempOutput);
      progress.setLabel("Encrypting archive");
      await encryptArchive(tempOutput, resolvedOutput, archivePassword as string);
    } finally {
      await fs.rm(tempOutput, { force: true }).catch(() => undefined);
    }
  });

  const stat = await fs.stat(path.resolve(outputPath));
  const exportedScopeList = scopes.join(", ");
  runtime.log(`Migration archive created: ${shortenHomePath(path.resolve(outputPath))}`);
  runtime.log(`Scopes: ${exportedScopeList}`);
  runtime.log(`Size: ${formatBytes(stat.size)}`);
}

export function parseMigrateExportOptions(raw: {
  scopes?: string;
  format?: string;
  password?: string;
  output?: string;
  yes?: boolean;
  nonInteractive?: boolean;
}): MigrateExportOptions {
  const format = raw.format?.trim();
  if (format && format !== "plain" && format !== "encrypted") {
    throw new Error('Invalid --format. Expected "plain" or "encrypted".');
  }
  return {
    scopes: parseMigrateScopesFlag(raw.scopes),
    format: format as MigrateFormat | undefined,
    password: raw.password?.trim() || undefined,
    output: raw.output?.trim() || undefined,
    yes: Boolean(raw.yes),
    nonInteractive: Boolean(raw.nonInteractive),
  };
}

async function collectMigrateArchiveEntries(
  scopes: readonly MigrateScope[],
): Promise<MigrateArchiveEntry[]> {
  const plan = resolveCleanupPlanFromDisk();
  const entries: MigrateArchiveEntry[] = [];

  if (scopes.includes("memory")) {
    const memoryDir = path.join(plan.stateDir, "memory", "soul");
    if (await pathExists(memoryDir)) {
      entries.push({
        scope: "memory",
        sourcePath: memoryDir,
        archivePath: resolveMigrateArchivePath({ scope: "memory", relativePath: "soul" }),
        kind: "directory",
      });
    }
  }

  if (scopes.includes("config") && (await pathExists(plan.configPath))) {
    entries.push({
      scope: "config",
      sourcePath: plan.configPath,
      archivePath: resolveMigrateArchivePath({
        scope: "config",
        relativePath: path.basename(plan.configPath),
      }),
      kind: "file",
    });
  }

  if (scopes.includes("sessions")) {
    const agentsRoot = path.join(plan.stateDir, "agents");
    try {
      const agents = await fs.readdir(agentsRoot, { withFileTypes: true });
      for (const agent of agents) {
        if (!agent.isDirectory()) {
          continue;
        }
        const agentId = agent.name;
        const sessionsDir = resolveSessionTranscriptsDirForAgent(agentId);
        if (await pathExists(sessionsDir)) {
          entries.push({
            scope: "sessions",
            sourcePath: sessionsDir,
            archivePath: resolveMigrateArchivePath({
              scope: "sessions",
              relativePath: `agents/${agentId}/sessions`,
            }),
            kind: "directory",
          });
        }
        const sessionStorePath = path.join(path.dirname(sessionsDir), "sessions.json");
        if (await pathExists(sessionStorePath)) {
          entries.push({
            scope: "sessions",
            sourcePath: sessionStorePath,
            archivePath: resolveMigrateArchivePath({
              scope: "sessions",
              relativePath: `agents/${agentId}/sessions.json`,
            }),
            kind: "file",
          });
        }
      }
    } catch {
      // No agents directory yet.
    }
  }

  if (scopes.includes("credentials") && (await pathExists(plan.oauthDir))) {
    entries.push({
      scope: "credentials",
      sourcePath: plan.oauthDir,
      archivePath: resolveMigrateArchivePath({ scope: "credentials" }),
      kind: "directory",
    });
  }

  if (scopes.includes("workspace")) {
    for (const [index, workspaceDir] of plan.workspaceDirs.entries()) {
      if (!(await pathExists(workspaceDir))) {
        continue;
      }
      entries.push({
        scope: "workspace",
        sourcePath: workspaceDir,
        archivePath: resolveMigrateArchivePath({
          scope: "workspace",
          workspaceSlot: index,
        }),
        kind: "directory",
      });
    }
  }

  if (scopes.includes("tasks")) {
    const tasksDir = path.join(plan.stateDir, "tasks");
    if (await pathExists(tasksDir)) {
      entries.push({
        scope: "tasks",
        sourcePath: tasksDir,
        archivePath: resolveMigrateArchivePath({ scope: "tasks" }),
        kind: "directory",
      });
    }
  }

  if (scopes.includes("ledger")) {
    const ledgerPath = resolveLedgerDbPath();
    if (await pathExists(ledgerPath)) {
      entries.push({
        scope: "ledger",
        sourcePath: ledgerPath,
        archivePath: resolveMigrateArchivePath({
          scope: "ledger",
          relativePath: "events.sqlite",
        }),
        kind: "file",
      });
    }
  }

  return entries;
}

async function pathExists(target: string): Promise<boolean> {
  try {
    await fs.lstat(target);
    return true;
  } catch {
    return false;
  }
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)}GB`;
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  }
  if (bytes >= 1024) {
    return `${Math.round(bytes / 1024)}KB`;
  }
  return `${bytes}B`;
}

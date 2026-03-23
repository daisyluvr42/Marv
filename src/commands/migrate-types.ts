import os from "node:os";
import path from "node:path";
import { resolveHomeDir } from "../utils.js";

export const MIGRATE_MANIFEST_VERSION = 1;
export const MIGRATE_SCOPES = [
  "memory",
  "config",
  "sessions",
  "credentials",
  "workspace",
  "tasks",
  "ledger",
] as const;
export const MIGRATE_DEFAULT_EXPORT_SCOPES = [
  "memory",
  "config",
  "sessions",
  "workspace",
] as const satisfies readonly MigrateScope[];
export const MIGRATE_ARCHIVE_ROOT = "scopes";

export type MigrateScope = (typeof MIGRATE_SCOPES)[number];
export type MigrateFormat = "plain" | "encrypted";
export type MigrateArchiveEntryKind = "file" | "directory";

export type MigrateExportOptions = {
  scopes?: MigrateScope[];
  format?: MigrateFormat;
  password?: string;
  output?: string;
  yes?: boolean;
  nonInteractive?: boolean;
};

export type MigrateImportOptions = {
  archivePath: string;
  scopes?: MigrateScope[];
  password?: string;
  force?: boolean;
  dryRun?: boolean;
  yes?: boolean;
  nonInteractive?: boolean;
};

export type MigrateArchiveEntry = {
  scope: MigrateScope;
  sourcePath: string;
  archivePath: string;
  kind: MigrateArchiveEntryKind;
};

export type MigrateManifestItem = Omit<MigrateArchiveEntry, "sourcePath">;

export type MigrateManifest = {
  version: typeof MIGRATE_MANIFEST_VERSION;
  exportedAt: string;
  marvVersion: string;
  scopes: MigrateScope[];
  format: MigrateFormat;
  platform: NodeJS.Platform;
  hostname: string;
  items: MigrateManifestItem[];
};

export const SCOPE_PATHS: Record<MigrateScope, readonly string[]> = {
  memory: ["memory/soul", "soul", "experience"],
  config: ["marv.json"],
  sessions: ["agents/*/sessions", "agents/*/sessions.json"],
  credentials: ["credentials"],
  workspace: ["workspace"],
  tasks: ["tasks"],
  ledger: ["ledger/events.sqlite"],
};

export const SCOPE_OPTIONS: Array<{
  value: MigrateScope;
  label: string;
  hint: string;
}> = [
  {
    value: "memory",
    label: "Memory",
    hint: "vector DBs, soul identity, and experience files",
  },
  {
    value: "config",
    label: "Config",
    hint: "marv.json",
  },
  {
    value: "sessions",
    label: "Sessions",
    hint: "agent transcripts and sessions.json",
  },
  {
    value: "credentials",
    label: "Credentials",
    hint: "API keys and channel auth",
  },
  {
    value: "workspace",
    label: "Workspace",
    hint: "agent files and notes",
  },
  {
    value: "tasks",
    label: "Tasks",
    hint: "task-context state",
  },
  {
    value: "ledger",
    label: "Ledger",
    hint: "ledger/events.sqlite",
  },
];

export function isMigrateScope(value: string): value is MigrateScope {
  return MIGRATE_SCOPES.includes(value as MigrateScope);
}

export function normalizeMigrateScopes(values: Iterable<string>): MigrateScope[] {
  const scopes: MigrateScope[] = [];
  const seen = new Set<MigrateScope>();
  for (const raw of values) {
    const scope = String(raw ?? "")
      .trim()
      .toLowerCase();
    if (!scope) {
      continue;
    }
    if (!isMigrateScope(scope)) {
      throw new Error(`Invalid scope: ${raw}`);
    }
    if (seen.has(scope)) {
      continue;
    }
    seen.add(scope);
    scopes.push(scope);
  }
  return scopes;
}

export function parseMigrateScopesFlag(raw: string | undefined): MigrateScope[] | undefined {
  const trimmed = raw?.trim();
  if (!trimmed) {
    return undefined;
  }
  return normalizeMigrateScopes(trimmed.split(","));
}

export function resolveMigrateArchivePath(params: {
  scope: MigrateScope;
  relativePath?: string;
  workspaceSlot?: number;
}): string {
  const relative = String(params.relativePath ?? "")
    .replaceAll("\\", "/")
    .replace(/^\/+/, "")
    .replace(/\/+$/, "");
  if (params.scope === "workspace") {
    const slot = params.workspaceSlot;
    if (!Number.isInteger(slot) || (slot as number) < 0) {
      throw new Error("Workspace exports require a workspace slot.");
    }
    return relative
      ? `${MIGRATE_ARCHIVE_ROOT}/workspace/${String(slot)}/${relative}`
      : `${MIGRATE_ARCHIVE_ROOT}/workspace/${String(slot)}`;
  }
  return relative
    ? `${MIGRATE_ARCHIVE_ROOT}/${params.scope}/${relative}`
    : `${MIGRATE_ARCHIVE_ROOT}/${params.scope}`;
}

export function getDefaultMigrateExportPath(format: MigrateFormat): string {
  const home = resolveHomeDir() ?? os.homedir();
  const date = new Date().toISOString().slice(0, 10);
  const fileName =
    format === "encrypted" ? `marv-export-${date}.tar.gz.enc` : `marv-export-${date}.tar.gz`;
  return path.join(home, fileName);
}

export function createMigrateManifest(params: {
  scopes: MigrateScope[];
  format: MigrateFormat;
  marvVersion: string;
  items: MigrateManifestItem[];
}): MigrateManifest {
  return {
    version: MIGRATE_MANIFEST_VERSION,
    exportedAt: new Date().toISOString(),
    marvVersion: params.marvVersion,
    scopes: [...params.scopes],
    format: params.format,
    platform: process.platform,
    hostname: os.hostname(),
    items: params.items.map((item) => ({ ...item })),
  };
}

export function parseMigrateManifest(value: unknown): MigrateManifest {
  if (!value || typeof value !== "object") {
    throw new Error("Invalid migration manifest.");
  }
  const raw = value as Partial<MigrateManifest> & { items?: unknown };
  if (raw.version !== MIGRATE_MANIFEST_VERSION) {
    throw new Error(`Unsupported migration manifest version: ${String(raw.version)}`);
  }
  if (typeof raw.exportedAt !== "string" || !raw.exportedAt.trim()) {
    throw new Error("Migration manifest is missing exportedAt.");
  }
  if (typeof raw.marvVersion !== "string" || !raw.marvVersion.trim()) {
    throw new Error("Migration manifest is missing marvVersion.");
  }
  if (raw.format !== "plain" && raw.format !== "encrypted") {
    throw new Error("Migration manifest has an invalid format.");
  }
  if (typeof raw.platform !== "string" || !raw.platform.trim()) {
    throw new Error("Migration manifest is missing platform.");
  }
  if (typeof raw.hostname !== "string" || !raw.hostname.trim()) {
    throw new Error("Migration manifest is missing hostname.");
  }
  if (!Array.isArray(raw.scopes)) {
    throw new Error("Migration manifest is missing scopes.");
  }
  const scopes = normalizeMigrateScopes(raw.scopes);
  if (!Array.isArray(raw.items)) {
    throw new Error("Migration manifest is missing items.");
  }
  const items = raw.items.map((item) => parseMigrateManifestItem(item));
  return {
    version: MIGRATE_MANIFEST_VERSION,
    exportedAt: raw.exportedAt,
    marvVersion: raw.marvVersion,
    scopes,
    format: raw.format,
    platform: raw.platform,
    hostname: raw.hostname,
    items,
  };
}

function parseMigrateManifestItem(value: unknown): MigrateManifestItem {
  if (!value || typeof value !== "object") {
    throw new Error("Migration manifest item is invalid.");
  }
  const raw = value as Partial<MigrateManifestItem>;
  if (typeof raw.archivePath !== "string" || !raw.archivePath.trim()) {
    throw new Error("Migration manifest item is missing archivePath.");
  }
  if (!raw.scope || !isMigrateScope(raw.scope)) {
    throw new Error(`Migration manifest item has invalid scope: ${String(raw.scope)}`);
  }
  if (raw.kind !== "file" && raw.kind !== "directory") {
    throw new Error("Migration manifest item has invalid kind.");
  }
  return {
    scope: raw.scope,
    archivePath: raw.archivePath,
    kind: raw.kind,
  };
}

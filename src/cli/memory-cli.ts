import fsSync from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { Command } from "commander";
import { resolveDefaultAgentId } from "../agents/agent-scope.js";
import { loadConfig } from "../core/config/config.js";
import { resolveStateDir } from "../core/config/paths.js";
import { resolveSessionTranscriptsDirForAgent } from "../core/config/sessions/paths.js";
import { setVerbose } from "../globals.js";
import { getMemorySearchManager, type MemorySearchManagerResult } from "../memory/index.js";
import { listMemoryFiles, normalizeExtraMemoryPaths } from "../memory/internal.js";
import {
  countSoulArchiveEvents,
  countSoulMemoryItemsByRecordKind,
  listSoulMemoryItems,
  querySoulMemoryMulti,
  resolveSoulMemoryDbPath,
  type SoulMemoryQueryResult,
  type SoulMemoryScope,
} from "../memory/storage/soul-memory-store.js";
import { defaultRuntime } from "../runtime.js";
import { formatDocsLink } from "../terminal/links.js";
import { colorize, isRich, theme } from "../terminal/theme.js";
import { shortenHomeInString, shortenHomePath } from "../utils.js";
import { formatErrorMessage, withManager } from "./cli-utils.js";
import { defineCommandPolicies } from "./command-policy.js";
import { formatHelpExamples } from "./help-format.js";
import { withProgress, withProgressTotals } from "./progress.js";

export const MEMORY_CLI_COMMAND_POLICIES = defineCommandPolicies("memory", [
  {
    path: "status",
    cliBootstrap: "skip",
    sideEffect: "none",
  },
]);

type MemoryCommandOptions = {
  agent?: string;
  json?: boolean;
  deep?: boolean;
  index?: boolean;
  force?: boolean;
  verbose?: boolean;
};

type MemoryManager = NonNullable<MemorySearchManagerResult["manager"]>;
type MemoryManagerPurpose = Parameters<typeof getMemorySearchManager>[0]["purpose"];

type MemorySourceName = "memory" | "sessions";

type SourceScan = {
  source: MemorySourceName;
  totalFiles: number | null;
  issues: string[];
};

type MemorySourceScan = {
  sources: SourceScan[];
  totalFiles: number | null;
  issues: string[];
};

function formatSourceLabel(source: string, workspaceDir: string, agentId: string): string {
  if (source === "memory") {
    return shortenHomeInString(
      `memory (MEMORY.md + ${path.join(workspaceDir, "memory")}${path.sep}*.md)`,
    );
  }
  if (source === "sessions") {
    const stateDir = resolveStateDir(process.env, os.homedir);
    return shortenHomeInString(
      `sessions (${path.join(stateDir, "agents", agentId, "sessions")}${path.sep}*.jsonl)`,
    );
  }
  return source;
}

function resolveAgent(cfg: ReturnType<typeof loadConfig>, agent?: string) {
  const trimmed = agent?.trim();
  if (trimmed) {
    return trimmed;
  }
  return resolveDefaultAgentId(cfg);
}

function resolveAgentIds(cfg: ReturnType<typeof loadConfig>, agent?: string): string[] {
  const trimmed = agent?.trim();
  if (trimmed) {
    return [trimmed];
  }
  return [resolveDefaultAgentId(cfg)];
}

function formatExtraPaths(workspaceDir: string, extraPaths: string[]): string[] {
  return normalizeExtraMemoryPaths(workspaceDir, extraPaths).map((entry) => shortenHomePath(entry));
}

async function withMemoryManagerForAgent(params: {
  cfg: ReturnType<typeof loadConfig>;
  agentId: string;
  purpose?: MemoryManagerPurpose;
  run: (manager: MemoryManager) => Promise<void>;
}): Promise<void> {
  const managerParams: Parameters<typeof getMemorySearchManager>[0] = {
    cfg: params.cfg,
    agentId: params.agentId,
  };
  if (params.purpose) {
    managerParams.purpose = params.purpose;
  }
  await withManager<MemoryManager>({
    getManager: () => getMemorySearchManager(managerParams),
    onMissing: (error) => defaultRuntime.log(error ?? "Memory search disabled."),
    onCloseError: (err) =>
      defaultRuntime.error(`Memory manager close failed: ${formatErrorMessage(err)}`),
    close: async (manager) => {
      await manager.close?.();
    },
    run: params.run,
  });
}

async function checkReadableFile(pathname: string): Promise<{ exists: boolean; issue?: string }> {
  try {
    await fs.access(pathname, fsSync.constants.R_OK);
    return { exists: true };
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      return { exists: false };
    }
    return {
      exists: true,
      issue: `${shortenHomePath(pathname)} not readable (${code ?? "error"})`,
    };
  }
}

async function scanSessionFiles(agentId: string): Promise<SourceScan> {
  const issues: string[] = [];
  const sessionsDir = resolveSessionTranscriptsDirForAgent(agentId);
  try {
    const entries = await fs.readdir(sessionsDir, { withFileTypes: true });
    const totalFiles = entries.filter(
      (entry) => entry.isFile() && entry.name.endsWith(".jsonl"),
    ).length;
    return { source: "sessions", totalFiles, issues };
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      issues.push(`sessions directory missing (${shortenHomePath(sessionsDir)})`);
      return { source: "sessions", totalFiles: 0, issues };
    }
    issues.push(
      `sessions directory not accessible (${shortenHomePath(sessionsDir)}): ${code ?? "error"}`,
    );
    return { source: "sessions", totalFiles: null, issues };
  }
}

async function scanMemoryFiles(
  workspaceDir: string,
  extraPaths: string[] = [],
): Promise<SourceScan> {
  const issues: string[] = [];
  const memoryFile = path.join(workspaceDir, "MEMORY.md");
  const altMemoryFile = path.join(workspaceDir, "memory.md");
  const memoryDir = path.join(workspaceDir, "memory");

  const primary = await checkReadableFile(memoryFile);
  const alt = await checkReadableFile(altMemoryFile);
  if (primary.issue) {
    issues.push(primary.issue);
  }
  if (alt.issue) {
    issues.push(alt.issue);
  }

  const resolvedExtraPaths = normalizeExtraMemoryPaths(workspaceDir, extraPaths);
  for (const extraPath of resolvedExtraPaths) {
    try {
      const stat = await fs.lstat(extraPath);
      if (stat.isSymbolicLink()) {
        continue;
      }
      const extraCheck = await checkReadableFile(extraPath);
      if (extraCheck.issue) {
        issues.push(extraCheck.issue);
      }
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (code === "ENOENT") {
        issues.push(`additional memory path missing (${shortenHomePath(extraPath)})`);
      } else {
        issues.push(
          `additional memory path not accessible (${shortenHomePath(extraPath)}): ${code ?? "error"}`,
        );
      }
    }
  }

  let dirReadable: boolean | null = null;
  try {
    await fs.access(memoryDir, fsSync.constants.R_OK);
    dirReadable = true;
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      issues.push(`memory directory missing (${shortenHomePath(memoryDir)})`);
      dirReadable = false;
    } else {
      issues.push(
        `memory directory not accessible (${shortenHomePath(memoryDir)}): ${code ?? "error"}`,
      );
      dirReadable = null;
    }
  }

  let listed: string[] = [];
  let listedOk = false;
  try {
    listed = await listMemoryFiles(workspaceDir, resolvedExtraPaths);
    listedOk = true;
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (dirReadable !== null) {
      issues.push(
        `memory directory scan failed (${shortenHomePath(memoryDir)}): ${code ?? "error"}`,
      );
      dirReadable = null;
    }
  }

  let totalFiles: number | null = 0;
  if (dirReadable === null) {
    totalFiles = null;
  } else {
    const files = new Set<string>(listedOk ? listed : []);
    if (!listedOk) {
      if (primary.exists) {
        files.add(memoryFile);
      }
      if (alt.exists) {
        files.add(altMemoryFile);
      }
    }
    totalFiles = files.size;
  }

  if ((totalFiles ?? 0) === 0 && issues.length === 0) {
    issues.push(`no memory files found in ${shortenHomePath(workspaceDir)}`);
  }

  return { source: "memory", totalFiles, issues };
}

async function summarizeQmdIndexArtifact(manager: MemoryManager): Promise<string | null> {
  const status = manager.status?.();
  if (!status || status.backend !== "qmd") {
    return null;
  }
  const dbPath = status.dbPath?.trim();
  if (!dbPath) {
    return null;
  }
  let stat: fsSync.Stats;
  try {
    stat = await fs.stat(dbPath);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === "ENOENT") {
      throw new Error(`QMD index file not found: ${shortenHomePath(dbPath)}`, { cause: err });
    }
    throw new Error(
      `QMD index file check failed: ${shortenHomePath(dbPath)} (${code ?? "error"})`,
      { cause: err },
    );
  }
  if (!stat.isFile() || stat.size <= 0) {
    throw new Error(`QMD index file is empty: ${shortenHomePath(dbPath)}`);
  }
  return `QMD index: ${shortenHomePath(dbPath)} (${stat.size} bytes)`;
}

async function scanMemorySources(params: {
  workspaceDir: string;
  agentId: string;
  sources: MemorySourceName[];
  extraPaths?: string[];
}): Promise<MemorySourceScan> {
  const scans: SourceScan[] = [];
  const extraPaths = params.extraPaths ?? [];
  for (const source of params.sources) {
    if (source === "memory") {
      scans.push(await scanMemoryFiles(params.workspaceDir, extraPaths));
    }
    if (source === "sessions") {
      scans.push(await scanSessionFiles(params.agentId));
    }
  }
  const issues = scans.flatMap((scan) => scan.issues);
  const totals = scans.map((scan) => scan.totalFiles);
  const numericTotals = totals.filter((total): total is number => total !== null);
  const totalFiles = totals.some((total) => total === null)
    ? null
    : numericTotals.reduce((sum, total) => sum + total, 0);
  return { sources: scans, totalFiles, issues };
}

export async function runMemoryStatus(opts: MemoryCommandOptions) {
  setVerbose(Boolean(opts.verbose));
  const cfg = loadConfig();
  const agentIds = resolveAgentIds(cfg, opts.agent);
  const allResults: Array<{
    agentId: string;
    status: ReturnType<MemoryManager["status"]>;
    embeddingProbe?: Awaited<ReturnType<MemoryManager["probeEmbeddingAvailability"]>>;
    indexError?: string;
    scan?: MemorySourceScan;
  }> = [];

  for (const agentId of agentIds) {
    const managerPurpose = opts.index ? "default" : "status";
    await withMemoryManagerForAgent({
      cfg,
      agentId,
      purpose: managerPurpose,
      run: async (manager) => {
        const deep = Boolean(opts.deep || opts.index);
        let embeddingProbe:
          | Awaited<ReturnType<typeof manager.probeEmbeddingAvailability>>
          | undefined;
        let indexError: string | undefined;
        const syncFn = manager.sync ? manager.sync.bind(manager) : undefined;
        if (deep) {
          await withProgress({ label: "Checking memory…", total: 2 }, async (progress) => {
            progress.setLabel("Probing vector…");
            await manager.probeVectorAvailability();
            progress.tick();
            progress.setLabel("Probing embeddings…");
            embeddingProbe = await manager.probeEmbeddingAvailability();
            progress.tick();
          });
          if (opts.index && syncFn) {
            await withProgressTotals(
              {
                label: "Indexing memory…",
                total: 0,
                fallback: opts.verbose ? "line" : undefined,
              },
              async (update, progress) => {
                try {
                  await syncFn({
                    reason: "cli",
                    force: Boolean(opts.force),
                    progress: (syncUpdate) => {
                      update({
                        completed: syncUpdate.completed,
                        total: syncUpdate.total,
                        label: syncUpdate.label,
                      });
                      if (syncUpdate.label) {
                        progress.setLabel(syncUpdate.label);
                      }
                    },
                  });
                } catch (err) {
                  indexError = formatErrorMessage(err);
                  defaultRuntime.error(`Memory index failed: ${indexError}`);
                  process.exitCode = 1;
                }
              },
            );
          } else if (opts.index && !syncFn) {
            defaultRuntime.log("Memory backend does not support manual reindex.");
          }
        } else {
          await manager.probeVectorAvailability();
        }
        const status = manager.status();
        const sources = (
          status.sources?.length ? status.sources : ["memory"]
        ) as MemorySourceName[];
        const workspaceDir = status.workspaceDir;
        const scan = workspaceDir
          ? await scanMemorySources({
              workspaceDir,
              agentId,
              sources,
              extraPaths: status.extraPaths,
            })
          : undefined;
        allResults.push({ agentId, status, embeddingProbe, indexError, scan });
      },
    });
  }

  if (opts.json) {
    defaultRuntime.log(JSON.stringify(allResults, null, 2));
    return;
  }

  const rich = isRich();
  const heading = (text: string) => colorize(rich, theme.heading, text);
  const muted = (text: string) => colorize(rich, theme.muted, text);
  const info = (text: string) => colorize(rich, theme.info, text);
  const success = (text: string) => colorize(rich, theme.success, text);
  const warn = (text: string) => colorize(rich, theme.warn, text);
  const accent = (text: string) => colorize(rich, theme.accent, text);
  const label = (text: string) => muted(`${text}:`);

  for (const result of allResults) {
    const { agentId, status, embeddingProbe, indexError, scan } = result;
    const filesIndexed = status.files ?? 0;
    const chunksIndexed = status.chunks ?? 0;
    const totalFiles = scan?.totalFiles ?? null;
    const indexedLabel =
      totalFiles === null
        ? `${filesIndexed}/? files · ${chunksIndexed} chunks`
        : `${filesIndexed}/${totalFiles} files · ${chunksIndexed} chunks`;
    if (opts.index) {
      const line = indexError ? `Memory index failed: ${indexError}` : "Memory index complete.";
      defaultRuntime.log(line);
    }
    const requestedProvider = status.requestedProvider ?? status.provider;
    const modelLabel = status.model ?? status.provider;
    const storePath = status.dbPath ? shortenHomePath(status.dbPath) : "<unknown>";
    const workspacePath = status.workspaceDir ? shortenHomePath(status.workspaceDir) : "<unknown>";
    const sourceList = status.sources?.length ? status.sources.join(", ") : null;
    const extraPaths = status.workspaceDir
      ? formatExtraPaths(status.workspaceDir, status.extraPaths ?? [])
      : [];
    const lines = [
      `${heading("Memory Search")} ${muted(`(${agentId})`)}`,
      `${label("Provider")} ${info(status.provider)} ${muted(`(requested: ${requestedProvider})`)}`,
      `${label("Model")} ${info(modelLabel)}`,
      sourceList ? `${label("Sources")} ${info(sourceList)}` : null,
      extraPaths.length ? `${label("Extra paths")} ${info(extraPaths.join(", "))}` : null,
      `${label("Indexed")} ${success(indexedLabel)}`,
      `${label("Dirty")} ${status.dirty ? warn("yes") : muted("no")}`,
      `${label("Store")} ${info(storePath)}`,
      `${label("Workspace")} ${info(workspacePath)}`,
    ].filter(Boolean) as string[];
    if (embeddingProbe) {
      const state = embeddingProbe.ok ? "ready" : "unavailable";
      const stateColor = embeddingProbe.ok ? theme.success : theme.warn;
      lines.push(`${label("Embeddings")} ${colorize(rich, stateColor, state)}`);
      if (embeddingProbe.error) {
        lines.push(`${label("Embeddings error")} ${warn(embeddingProbe.error)}`);
      }
    }
    if (status.sourceCounts?.length) {
      lines.push(label("By source"));
      for (const entry of status.sourceCounts) {
        const total = scan?.sources?.find(
          (scanEntry) => scanEntry.source === entry.source,
        )?.totalFiles;
        const counts =
          total === null
            ? `${entry.files}/? files · ${entry.chunks} chunks`
            : `${entry.files}/${total} files · ${entry.chunks} chunks`;
        lines.push(`  ${accent(entry.source)} ${muted("·")} ${muted(counts)}`);
      }
    }
    if (status.fallback) {
      lines.push(`${label("Fallback")} ${warn(status.fallback.from)}`);
    }
    if (status.vector) {
      const vectorState = status.vector.enabled
        ? status.vector.available === undefined
          ? "unknown"
          : status.vector.available
            ? "ready"
            : "unavailable"
        : "disabled";
      const vectorColor =
        vectorState === "ready"
          ? theme.success
          : vectorState === "unavailable"
            ? theme.warn
            : theme.muted;
      lines.push(`${label("Vector")} ${colorize(rich, vectorColor, vectorState)}`);
      if (status.vector.dims) {
        lines.push(`${label("Vector dims")} ${info(String(status.vector.dims))}`);
      }
      if (status.vector.extensionPath) {
        lines.push(`${label("Vector path")} ${info(shortenHomePath(status.vector.extensionPath))}`);
      }
      if (status.vector.loadError) {
        lines.push(`${label("Vector error")} ${warn(status.vector.loadError)}`);
      }
    }
    if (status.fts) {
      const ftsState = status.fts.enabled
        ? status.fts.available
          ? "ready"
          : "unavailable"
        : "disabled";
      const ftsColor =
        ftsState === "ready"
          ? theme.success
          : ftsState === "unavailable"
            ? theme.warn
            : theme.muted;
      lines.push(`${label("FTS")} ${colorize(rich, ftsColor, ftsState)}`);
      if (status.fts.error) {
        lines.push(`${label("FTS error")} ${warn(status.fts.error)}`);
      }
    }
    if (status.cache) {
      const cacheState = status.cache.enabled ? "enabled" : "disabled";
      const cacheColor = status.cache.enabled ? theme.success : theme.muted;
      const suffix =
        status.cache.enabled && typeof status.cache.entries === "number"
          ? ` (${status.cache.entries} entries)`
          : "";
      lines.push(`${label("Embedding cache")} ${colorize(rich, cacheColor, cacheState)}${suffix}`);
      if (status.cache.enabled && typeof status.cache.maxEntries === "number") {
        lines.push(`${label("Cache cap")} ${info(String(status.cache.maxEntries))}`);
      }
    }
    if (status.batch) {
      const batchState = status.batch.enabled ? "enabled" : "disabled";
      const batchColor = status.batch.enabled ? theme.success : theme.warn;
      const batchSuffix = ` (failures ${status.batch.failures}/${status.batch.limit})`;
      lines.push(
        `${label("Batch")} ${colorize(rich, batchColor, batchState)}${muted(batchSuffix)}`,
      );
      if (status.batch.lastError) {
        lines.push(`${label("Batch error")} ${warn(status.batch.lastError)}`);
      }
    }
    if (status.fallback?.reason) {
      lines.push(muted(status.fallback.reason));
    }
    if (indexError) {
      lines.push(`${label("Index error")} ${warn(indexError)}`);
    }
    if (scan?.issues.length) {
      lines.push(label("Issues"));
      for (const issue of scan.issues) {
        lines.push(`  ${warn(issue)}`);
      }
    }
    const recordCounts = countSoulMemoryItemsByRecordKind({ agentId });
    const totalItems =
      recordCounts.fact + recordCounts.relationship + recordCounts.experience + recordCounts.soul;
    const archiveCount = countSoulArchiveEvents({ agentId });
    lines.push(`${label("Structured")} ${info(`${totalItems} items · Archive ${archiveCount}`)}`);
    lines.push(
      `${label("Kinds")} ${info(
        `fact ${recordCounts.fact} · relationship ${recordCounts.relationship} · experience ${recordCounts.experience} · soul ${recordCounts.soul}`,
      )}`,
    );
    defaultRuntime.log(lines.join("\n"));
    defaultRuntime.log("");
  }
}

export function registerMemoryCli(program: Command) {
  const memory = program
    .command("mem")
    .alias("memory")
    .enablePositionalOptions()
    .description("Search, inspect, and reindex memory files")
    .addHelpText(
      "after",
      () =>
        `\n${theme.heading("Examples:")}\n${formatHelpExamples([
          ["marv mem list", "Browse all structured memory items."],
          ["marv mem backup", "Back up memory database to a file."],
          ["marv mem restore backup.sqlite", "Restore memory from a backup."],
          ["marv mem export", "Export memory items to JSON."],
          ["marv mem import export.json", "Import memory items from JSON."],
          ["marv memory status", "Show index and provider status."],
          ['marv memory search "deployment notes"', "Search indexed memory entries."],
        ])}\n\n${theme.muted("Docs:")} ${formatDocsLink("/cli/memory", "docs: /cli/memory")}\n`,
    );

  // P0 subcommand (core memory sections)
  const p0 = memory
    .command("p0")
    .description("Manage P0 core memory sections (soul, identity, user)")
    .option("--json", "Print JSON output");

  p0.action(async (opts: { json?: boolean }) => {
    const { memoryP0ShowCommand } = await import("../commands/memory-p0.js");
    await memoryP0ShowCommand(opts, defaultRuntime);
  });

  for (const section of ["soul", "identity", "user"] as const) {
    p0.command(section)
      .description(`View or set the ${section} section`)
      .argument("[value]", `New value for ${section}`)
      .option("--json", "Print JSON output")
      .option("--file <path>", "Read value from file")
      .option("--clear", "Clear the section")
      .action(
        async (
          value: string | undefined,
          opts: { json?: boolean; file?: string; clear?: boolean },
        ) => {
          // Inherit --json from parent p0 command if not set on subcommand
          const merged = { ...p0.opts(), ...opts };
          const { memoryP0SectionCommand } = await import("../commands/memory-p0.js");
          await memoryP0SectionCommand(section, value, merged, defaultRuntime);
        },
      );
  }

  memory
    .command("status")
    .description("Show memory search index status")
    .option("--agent <id>", "Agent id (default: default agent)")
    .option("--json", "Print JSON")
    .option("--deep", "Probe embedding provider availability")
    .option("--index", "Reindex if dirty (implies --deep)")
    .option("--verbose", "Verbose logging", false)
    .action(async (opts: MemoryCommandOptions & { force?: boolean }) => {
      await runMemoryStatus(opts);
    });

  memory
    .command("index")
    .description("Reindex memory files")
    .option("--agent <id>", "Agent id (default: default agent)")
    .option("--force", "Force full reindex", false)
    .option("--verbose", "Verbose logging", false)
    .action(async (opts: MemoryCommandOptions) => {
      setVerbose(Boolean(opts.verbose));
      const cfg = loadConfig();
      const agentIds = resolveAgentIds(cfg, opts.agent);
      for (const agentId of agentIds) {
        await withMemoryManagerForAgent({
          cfg,
          agentId,
          run: async (manager) => {
            try {
              const syncFn = manager.sync ? manager.sync.bind(manager) : undefined;
              if (opts.verbose) {
                const status = manager.status();
                const rich = isRich();
                const heading = (text: string) => colorize(rich, theme.heading, text);
                const muted = (text: string) => colorize(rich, theme.muted, text);
                const info = (text: string) => colorize(rich, theme.info, text);
                const warn = (text: string) => colorize(rich, theme.warn, text);
                const label = (text: string) => muted(`${text}:`);
                const sourceLabels = (status.sources ?? []).map((source) =>
                  formatSourceLabel(source, status.workspaceDir ?? "", agentId),
                );
                const extraPaths = status.workspaceDir
                  ? formatExtraPaths(status.workspaceDir, status.extraPaths ?? [])
                  : [];
                const requestedProvider = status.requestedProvider ?? status.provider;
                const modelLabel = status.model ?? status.provider;
                const lines = [
                  `${heading("Memory Index")} ${muted(`(${agentId})`)}`,
                  `${label("Provider")} ${info(status.provider)} ${muted(
                    `(requested: ${requestedProvider})`,
                  )}`,
                  `${label("Model")} ${info(modelLabel)}`,
                  sourceLabels.length
                    ? `${label("Sources")} ${info(sourceLabels.join(", "))}`
                    : null,
                  extraPaths.length
                    ? `${label("Extra paths")} ${info(extraPaths.join(", "))}`
                    : null,
                ].filter(Boolean) as string[];
                if (status.fallback) {
                  lines.push(`${label("Fallback")} ${warn(status.fallback.from)}`);
                }
                defaultRuntime.log(lines.join("\n"));
                defaultRuntime.log("");
              }
              const startedAt = Date.now();
              let lastLabel = "Indexing memory…";
              let lastCompleted = 0;
              let lastTotal = 0;
              const formatElapsed = () => {
                const elapsedMs = Math.max(0, Date.now() - startedAt);
                const seconds = Math.floor(elapsedMs / 1000);
                const minutes = Math.floor(seconds / 60);
                const remainingSeconds = seconds % 60;
                return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
              };
              const formatEta = () => {
                if (lastTotal <= 0 || lastCompleted <= 0) {
                  return null;
                }
                const elapsedMs = Math.max(1, Date.now() - startedAt);
                const rate = lastCompleted / elapsedMs;
                if (!Number.isFinite(rate) || rate <= 0) {
                  return null;
                }
                const remainingMs = Math.max(0, (lastTotal - lastCompleted) / rate);
                const seconds = Math.floor(remainingMs / 1000);
                const minutes = Math.floor(seconds / 60);
                const remainingSeconds = seconds % 60;
                return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
              };
              const buildLabel = () => {
                const elapsed = formatElapsed();
                const eta = formatEta();
                return eta
                  ? `${lastLabel} · elapsed ${elapsed} · eta ${eta}`
                  : `${lastLabel} · elapsed ${elapsed}`;
              };
              if (!syncFn) {
                defaultRuntime.log("Memory backend does not support manual reindex.");
                return;
              }
              await withProgressTotals(
                {
                  label: "Indexing memory…",
                  total: 0,
                  fallback: opts.verbose ? "line" : undefined,
                },
                async (update, progress) => {
                  const interval = setInterval(() => {
                    progress.setLabel(buildLabel());
                  }, 1000);
                  try {
                    await syncFn({
                      reason: "cli",
                      force: Boolean(opts.force),
                      progress: (syncUpdate) => {
                        if (syncUpdate.label) {
                          lastLabel = syncUpdate.label;
                        }
                        lastCompleted = syncUpdate.completed;
                        lastTotal = syncUpdate.total;
                        update({
                          completed: syncUpdate.completed,
                          total: syncUpdate.total,
                          label: buildLabel(),
                        });
                        progress.setLabel(buildLabel());
                      },
                    });
                  } finally {
                    clearInterval(interval);
                  }
                },
              );
              const qmdIndexSummary = await summarizeQmdIndexArtifact(manager);
              if (qmdIndexSummary) {
                defaultRuntime.log(qmdIndexSummary);
              }
              defaultRuntime.log(`Memory index updated (${agentId}).`);
            } catch (err) {
              const message = formatErrorMessage(err);
              defaultRuntime.error(`Memory index failed (${agentId}): ${message}`);
              process.exitCode = 1;
            }
          },
        });
      }
    });

  memory
    .command("search")
    .description("Search structured memory and indexed files")
    .argument("<query>", "Search query")
    .option("--agent <id>", "Agent id (default: default agent)")
    .option("--max-results <n>", "Max results", (value: string) => Number(value))
    .option("--min-score <n>", "Minimum score", (value: string) => Number(value))
    .option("--json", "Print JSON")
    .action(
      async (
        query: string,
        opts: MemoryCommandOptions & {
          maxResults?: number;
          minScore?: number;
        },
      ) => {
        const cfg = loadConfig();
        const agentId = resolveAgent(cfg, opts.agent);
        const maxResults = opts.maxResults ?? 10;
        const minScore = opts.minScore ?? 0.3;

        // 1. Query soul memory (structured items) — primary source
        const scopes: SoulMemoryScope[] = [{ scopeType: "agent", scopeId: agentId, weight: 1 }];
        let soulResults: SoulMemoryQueryResult[] = [];
        try {
          soulResults = querySoulMemoryMulti({
            agentId,
            scopes,
            query,
            topK: maxResults,
            minScore,
          });
        } catch {
          // Soul memory may not be initialized; continue to file-based search
        }

        // 2. Query file-based index as secondary source (if soul results insufficient)
        type FileResult = {
          score: number;
          path: string;
          startLine: number;
          endLine: number;
          snippet: string;
        };
        let fileResults: FileResult[] = [];
        if (soulResults.length < maxResults) {
          try {
            const managerResult = await getMemorySearchManager({ cfg, agentId });
            const manager = managerResult.manager;
            if (manager) {
              try {
                fileResults = await manager.search(query, {
                  maxResults,
                  minScore,
                });
              } catch {
                // File-based search may fail; that's ok if we have soul results
              } finally {
                await manager.close?.();
              }
            }
          } catch {
            // Manager creation may fail
          }
        }

        // 3. Merge and display results
        type MergedResult =
          | { source: "structured"; score: number; item: SoulMemoryQueryResult }
          | { source: "file"; score: number; file: FileResult };
        const merged: MergedResult[] = [
          ...soulResults.map((item) => ({
            source: "structured" as const,
            score: item.score,
            item,
          })),
          ...fileResults.map((file) => ({
            source: "file" as const,
            score: file.score,
            file,
          })),
        ]
          .toSorted((a, b) => b.score - a.score)
          .slice(0, maxResults);

        if (opts.json) {
          defaultRuntime.log(JSON.stringify({ results: merged }, null, 2));
          return;
        }
        if (merged.length === 0) {
          defaultRuntime.log("No matches.");
          return;
        }
        const rich = isRich();
        const lines: string[] = [];
        for (const result of merged) {
          if (result.source === "structured") {
            const item = result.item;
            const kindLabel = item.recordKind ?? item.kind;
            lines.push(
              `${colorize(rich, theme.success, item.score.toFixed(3))} ${colorize(
                rich,
                theme.accent,
                `[${kindLabel}]`,
              )} ${colorize(rich, theme.muted, `conf=${item.confidence.toFixed(2)}`)}`,
            );
            const preview =
              item.content.length > 200 ? `${item.content.slice(0, 200)}…` : item.content;
            lines.push(preview);
            lines.push("");
          } else {
            const file = result.file;
            lines.push(
              `${colorize(rich, theme.success, file.score.toFixed(3))} ${colorize(
                rich,
                theme.accent,
                `${shortenHomePath(file.path)}:${file.startLine}-${file.endLine}`,
              )}`,
            );
            lines.push(colorize(rich, theme.muted, file.snippet));
            lines.push("");
          }
        }
        defaultRuntime.log(lines.join("\n").trim());
      },
    );

  // ── marv mem export ──
  memory
    .command("export")
    .description("Export all structured memory items to a JSON file")
    .option("--agent <id>", "Agent id (default: default agent)")
    .option("--kind <kind>", "Filter by kind (fact, relationship, experience, soul)")
    .option("--output <path>", "Output file path (default: ~/marv-memory-<agent>-<date>.json)")
    .action(async (opts: { agent?: string; kind?: string; output?: string }) => {
      const cfg = loadConfig();
      const agentId = resolveAgent(cfg, opts.agent);
      const items = listSoulMemoryItems({
        agentId,
        recordKind: opts.kind?.trim().toLowerCase() as
          | "fact"
          | "relationship"
          | "experience"
          | "soul"
          | undefined,
        limit: 500,
      });
      const exportData = {
        version: 1,
        exportedAt: new Date().toISOString(),
        agentId,
        items: items.map((item) => ({
          kind: item.kind,
          content: item.content,
          recordKind: item.recordKind,
          source: item.source,
          confidence: item.confidence,
          createdAt: item.createdAt,
          metadata: item.metadata,
        })),
      };
      const date = new Date().toISOString().slice(0, 10);
      const defaultOutput = path.join(os.homedir(), `marv-memory-${agentId}-${date}.json`);
      const outputPath = path.resolve(opts.output?.trim() || defaultOutput);
      await fs.mkdir(path.dirname(outputPath), { recursive: true });
      await fs.writeFile(outputPath, JSON.stringify(exportData, null, 2), "utf-8");
      defaultRuntime.log(`Exported ${items.length} memory items to ${shortenHomePath(outputPath)}`);
    });

  // ── marv mem import ──
  memory
    .command("import <file>")
    .description("Import memory items from a JSON export file")
    .option("--agent <id>", "Agent id (default: default agent)")
    .action(async (file: string, opts: { agent?: string }) => {
      const cfg = loadConfig();
      const agentId = resolveAgent(cfg, opts.agent);
      const sourcePath = path.resolve(file);
      let raw: string;
      try {
        raw = await fs.readFile(sourcePath, "utf-8");
      } catch {
        defaultRuntime.error(`File not found: ${shortenHomePath(sourcePath)}`);
        process.exitCode = 1;
        return;
      }
      let exportData: {
        version?: number;
        items?: Array<{
          kind?: string;
          content?: string;
          tier?: string;
          recordKind?: string;
          source?: string;
          confidence?: number;
          metadata?: Record<string, unknown>;
        }>;
      };
      try {
        exportData = JSON.parse(raw);
      } catch {
        defaultRuntime.error("Invalid JSON file.");
        process.exitCode = 1;
        return;
      }
      if (!Array.isArray(exportData.items) || exportData.items.length === 0) {
        defaultRuntime.log("No items to import.");
        return;
      }
      const { writeSoulMemory } = await import("../memory/storage/soul-memory-store.js");
      let imported = 0;
      for (const entry of exportData.items) {
        const kind = entry.kind?.trim();
        const content = entry.content?.trim();
        if (!kind || !content) {
          continue;
        }
        const recordKind = entry.recordKind?.trim() || "experience";
        const result = writeSoulMemory({
          agentId,
          scopeType: "agent",
          scopeId: agentId,
          kind,
          content,
          source: "manual_log",
          recordKind: recordKind as "fact" | "relationship" | "experience" | "soul",
        });
        if (result) {
          imported += 1;
        }
      }
      defaultRuntime.log(
        `Imported ${imported}/${exportData.items.length} items into agent "${agentId}".`,
      );
    });

  // ── marv mem list ──
  memory
    .command("list")
    .description("List structured soul memory items")
    .option("--agent <id>", "Agent id (default: default agent)")
    .option("--kind <kind>", "Filter by kind (fact, relationship, experience, soul)")
    .option("--limit <n>", "Max items to show", (v: string) => Number(v))
    .option("--json", "Print JSON")
    .action(async (opts: { agent?: string; kind?: string; limit?: number; json?: boolean }) => {
      const cfg = loadConfig();
      const agentId = resolveAgent(cfg, opts.agent);
      const items = listSoulMemoryItems({
        agentId,
        recordKind: opts.kind?.trim().toLowerCase() as
          | "fact"
          | "relationship"
          | "experience"
          | "soul"
          | undefined,
        limit: opts.limit ?? 50,
      });
      if (opts.json) {
        defaultRuntime.log(JSON.stringify(items, null, 2));
        return;
      }
      if (items.length === 0) {
        defaultRuntime.log("No memory items found.");
        return;
      }
      const rich = isRich();
      const lines: string[] = [];
      for (const item of items) {
        const recordLabel = colorize(rich, theme.accent, item.recordKind);
        const kindLabel = colorize(rich, theme.muted, `[${item.kind}]`);
        const age = Math.floor((Date.now() - item.createdAt) / (24 * 60 * 60 * 1000));
        const ageLabel = colorize(rich, theme.muted, `${age}d ago`);
        const conf = colorize(rich, theme.muted, `conf=${item.confidence.toFixed(2)}`);
        lines.push(`${recordLabel} ${kindLabel} ${item.content}`);
        lines.push(colorize(rich, theme.muted, `  ${ageLabel} · ${conf}`));
      }
      defaultRuntime.log(lines.join("\n"));
      defaultRuntime.log(colorize(rich, theme.muted, `\n${items.length} item(s) shown.`));
    });

  // ── marv mem backup ──
  memory
    .command("backup")
    .description("Back up soul memory database to a file")
    .option("--agent <id>", "Agent id (default: default agent)")
    .option("--output <path>", "Output file path (default: ~/marv-memory-backup-<date>.sqlite)")
    .action(async (opts: { agent?: string; output?: string }) => {
      const cfg = loadConfig();
      const agentId = resolveAgent(cfg, opts.agent);
      const dbPath = resolveSoulMemoryDbPath(agentId);
      try {
        await fs.access(dbPath, fsSync.constants.R_OK);
      } catch {
        defaultRuntime.error(`No soul memory database found for agent "${agentId}".`);
        defaultRuntime.error(`Expected at: ${shortenHomePath(dbPath)}`);
        process.exitCode = 1;
        return;
      }
      const date = new Date().toISOString().slice(0, 10);
      const defaultOutput = path.join(os.homedir(), `marv-memory-backup-${agentId}-${date}.sqlite`);
      const outputPath = path.resolve(opts.output?.trim() || defaultOutput);
      await fs.mkdir(path.dirname(outputPath), { recursive: true });
      await fs.copyFile(dbPath, outputPath);
      const stat = await fs.stat(outputPath);
      const { countSoulMemoryItems } = await import("../memory/storage/soul-memory-store.js");
      const total = countSoulMemoryItems({ agentId });
      defaultRuntime.log(`Memory backed up: ${shortenHomePath(outputPath)}`);
      defaultRuntime.log(`Agent: ${agentId} · ${total} items`);
      defaultRuntime.log(`Size: ${formatBackupSize(stat.size)}`);
    });

  // ── marv mem restore ──
  memory
    .command("restore <file>")
    .description("Restore soul memory database from a backup file")
    .option("--agent <id>", "Agent id (default: default agent)")
    .option("--force", "Overwrite existing database without prompting", false)
    .action(async (file: string, opts: { agent?: string; force?: boolean }) => {
      const cfg = loadConfig();
      const agentId = resolveAgent(cfg, opts.agent);
      const sourcePath = path.resolve(file);
      try {
        await fs.access(sourcePath, fsSync.constants.R_OK);
      } catch {
        defaultRuntime.error(`Backup file not found: ${shortenHomePath(sourcePath)}`);
        process.exitCode = 1;
        return;
      }
      const dbPath = resolveSoulMemoryDbPath(agentId);
      let existingExists = false;
      try {
        await fs.access(dbPath);
        existingExists = true;
      } catch {
        // No existing DB — safe to restore.
      }
      if (existingExists && !opts.force) {
        const { countSoulMemoryItems } = await import("../memory/storage/soul-memory-store.js");
        const total = countSoulMemoryItems({ agentId });
        defaultRuntime.error(`Existing memory database has ${total} items.`);
        defaultRuntime.error("Use --force to overwrite.");
        process.exitCode = 1;
        return;
      }
      await fs.mkdir(path.dirname(dbPath), { recursive: true });
      await fs.copyFile(sourcePath, dbPath);
      const { countSoulMemoryItems: countItems } =
        await import("../memory/storage/soul-memory-store.js");
      const total = countItems({ agentId });
      defaultRuntime.log(`Memory restored for agent "${agentId}".`);
      defaultRuntime.log(`${total} items`);
    });

  // ── marv mem migrate-embeddings ──
  memory
    .command("migrate-embeddings")
    .description("Re-embed memory items with ML embedding provider")
    .option("--agent <id>", "Agent id (default: default agent)")
    .option("--batch-size <n>", "Items per batch", (v: string) => Number(v))
    .option("--all", "Process all legacy items (may be slow)", false)
    .action(async (opts: { agent?: string; batchSize?: number; all?: boolean }) => {
      const cfg = loadConfig();
      const agentId = resolveAgent(cfg, opts.agent);
      const batchSize = Math.max(1, Math.min(500, Math.trunc(opts.batchSize ?? 50)));
      const { reembedLegacyBatch } = await import("../memory/storage/soul-memory-ml-bridge.js");

      let totalProcessed = 0;
      const maxIterations = opts.all ? 100 : 1;
      for (let i = 0; i < maxIterations; i += 1) {
        const count = await reembedLegacyBatch({ cfg, agentId, batchSize });
        totalProcessed += count;
        if (count < batchSize) {
          break;
        }
        defaultRuntime.log(`  Batch ${i + 1}: re-embedded ${count} items`);
      }

      if (totalProcessed === 0) {
        defaultRuntime.log(
          "No items to re-embed. Either all items already have ML embeddings, or no ML embedding provider is configured.",
        );
      } else {
        defaultRuntime.log(`Re-embedded ${totalProcessed} items for agent "${agentId}".`);
      }
    });
}

function formatBackupSize(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  }
  if (bytes >= 1024) {
    return `${Math.round(bytes / 1024)}KB`;
  }
  return `${bytes}B`;
}

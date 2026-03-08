import fs from "node:fs/promises";
import path from "node:path";
import {
  listSoulMemoryItems,
  writeSoulMemory,
  type SoulMemoryItem,
  type SoulMemoryTier,
} from "./soul-memory-store.js";

const MIRROR_META_PREFIX = "<!-- marv-memory-mirror hash=";
const DEFAULT_MIRROR_FILE = "MEMORY.md";

type MirrorEntry = {
  tier: SoulMemoryTier;
  kind: string;
  content: string;
};

export type SoulMemoryMirrorConflict = {
  kind: string;
  tier: SoulMemoryTier;
  content: string;
};

export type SoulMemoryMirrorExportResult = {
  updated: boolean;
  path: string;
  text: string;
  conflicts: SoulMemoryMirrorConflict[];
  requiresConfirmation: boolean;
};

export type SoulMemoryMirrorImportResult = {
  path: string;
  imported: number;
  conflictsResolved: number;
};

export async function exportSoulMemoryMirror(params: {
  agentId: string;
  workspaceDir: string;
  force?: boolean;
}): Promise<SoulMemoryMirrorExportResult> {
  const targetPath = path.join(params.workspaceDir, DEFAULT_MIRROR_FILE);
  const entries = loadMirrorEntries(params.agentId);
  const hash = computeMirrorHash(entries);
  const text = renderMirrorDocument(entries, hash);

  const existing = await readMirrorFile(targetPath);
  const conflicts = existing ? detectUnsyncedMirrorChanges(existing.text, existing.hash) : [];
  if (conflicts.length > 0 && !params.force) {
    return {
      updated: false,
      path: targetPath,
      text,
      conflicts,
      requiresConfirmation: true,
    };
  }

  await fs.writeFile(targetPath, text, "utf-8");
  return {
    updated: true,
    path: targetPath,
    text,
    conflicts,
    requiresConfirmation: false,
  };
}

export async function importSoulMemoryMirror(params: {
  agentId: string;
  workspaceDir: string;
  nowMs?: number;
}): Promise<SoulMemoryMirrorImportResult> {
  const targetPath = path.join(params.workspaceDir, DEFAULT_MIRROR_FILE);
  const existing = await readMirrorFile(targetPath);
  if (!existing) {
    return {
      path: targetPath,
      imported: 0,
      conflictsResolved: 0,
    };
  }

  const entries = parseMirrorEntries(existing.text);
  let imported = 0;
  for (const entry of entries) {
    const item = writeSoulMemory({
      agentId: params.agentId,
      scopeType: "agent",
      scopeId: params.agentId,
      kind: entry.kind,
      content: entry.content,
      tier: entry.tier,
      source: "manual_log",
      recordKind: entry.tier === "P0" ? "soul" : "experience",
      nowMs: params.nowMs,
    });
    if (item) {
      imported += 1;
    }
  }

  await exportSoulMemoryMirror({
    agentId: params.agentId,
    workspaceDir: params.workspaceDir,
    force: true,
  });

  return {
    path: targetPath,
    imported,
    conflictsResolved:
      existing.hash && computeMirrorHash(entries) !== existing.hash ? entries.length : 0,
  };
}

function loadMirrorEntries(agentId: string): MirrorEntry[] {
  const p0 = listSoulMemoryItems({
    agentId,
    scopeType: "agent",
    scopeId: agentId,
    tier: "P0",
    limit: 200,
  });
  const p1 = listSoulMemoryItems({
    agentId,
    scopeType: "agent",
    scopeId: agentId,
    tier: "P1",
    limit: 200,
  }).filter((item) => isMirrorableP1(item));

  return [...toMirrorEntries(p0), ...toMirrorEntries(p1)].toSorted((a, b) => {
    if (a.tier !== b.tier) {
      return a.tier.localeCompare(b.tier);
    }
    if (a.kind !== b.kind) {
      return a.kind.localeCompare(b.kind);
    }
    return a.content.localeCompare(b.content);
  });
}

function isMirrorableP1(item: SoulMemoryItem): boolean {
  return item.recordKind === "soul" || item.kind === "preference" || item.kind === "policy";
}

function toMirrorEntries(items: SoulMemoryItem[]): MirrorEntry[] {
  return items.map((item) => ({
    tier: item.tier,
    kind: item.kind,
    content: item.content.trim(),
  }));
}

function renderMirrorDocument(entries: MirrorEntry[], hash: string): string {
  const p0 = entries.filter((entry) => entry.tier === "P0");
  const p1 = entries.filter((entry) => entry.tier === "P1");
  const lines = [
    `${MIRROR_META_PREFIX}${hash} -->`,
    "# Memory Mirror",
    "",
    "This file mirrors selected structured memory.",
    "Edits here are not authoritative until you explicitly import them back into structured memory.",
    "",
    "## P0",
    ...renderEntryLines(p0),
    "",
    "## Stable P1",
    ...renderEntryLines(p1),
    "",
  ];
  return `${lines.join("\n").trimEnd()}\n`;
}

function renderEntryLines(entries: MirrorEntry[]): string[] {
  if (entries.length === 0) {
    return ["- (empty)"];
  }
  return entries.map((entry) => `- [${entry.kind}] ${entry.content}`);
}

async function readMirrorFile(
  filePath: string,
): Promise<{ text: string; hash: string | null } | null> {
  try {
    const text = await fs.readFile(filePath, "utf-8");
    return {
      text,
      hash: parseMirrorHash(text),
    };
  } catch {
    return null;
  }
}

function parseMirrorHash(text: string): string | null {
  const firstLine = text.split("\n", 1)[0]?.trim() ?? "";
  if (!firstLine.startsWith(MIRROR_META_PREFIX) || !firstLine.endsWith("-->")) {
    return null;
  }
  return firstLine.slice(MIRROR_META_PREFIX.length, -4).trim() || null;
}

function parseMirrorEntries(text: string): MirrorEntry[] {
  const entries: MirrorEntry[] = [];
  let tier: SoulMemoryTier = "P0";
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (line === "## P0") {
      tier = "P0";
      continue;
    }
    if (line === "## Stable P1") {
      tier = "P1";
      continue;
    }
    const match = line.match(/^- \[([^\]]+)\] (.+)$/);
    if (!match) {
      continue;
    }
    const kind = match[1]?.trim().toLowerCase();
    const content = match[2]?.trim();
    if (!kind || !content || content === "(empty)") {
      continue;
    }
    entries.push({ tier, kind, content });
  }
  return entries;
}

function detectUnsyncedMirrorChanges(
  text: string,
  hash: string | null,
): SoulMemoryMirrorConflict[] {
  const entries = parseMirrorEntries(text);
  if (entries.length === 0) {
    return [];
  }
  const currentHash = computeMirrorHash(entries);
  if (hash && currentHash === hash) {
    return [];
  }
  return entries.map((entry) => ({
    kind: entry.kind,
    tier: entry.tier,
    content: entry.content,
  }));
}

function computeMirrorHash(entries: MirrorEntry[]): string {
  const canonical = JSON.stringify(
    entries.map((entry) => ({
      tier: entry.tier,
      kind: entry.kind,
      content: entry.content.replace(/\s+/g, " ").trim(),
    })),
  );
  let hash = 0;
  for (let i = 0; i < canonical.length; i += 1) {
    hash = (hash * 31 + canonical.charCodeAt(i)) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

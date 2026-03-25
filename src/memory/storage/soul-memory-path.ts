import os from "node:os";
import path from "node:path";
import { resolveStateDir } from "../../core/config/paths.js";
import {
  SOUL_ARCHIVE_PATH_PREFIX,
  SOUL_MEMORY_PATH_PREFIX,
  normalizeScopeValue,
} from "./soul-memory-types.js";

export function buildSoulMemoryPath(itemId: string): string {
  return `${SOUL_MEMORY_PATH_PREFIX}${itemId}`;
}

export function buildSoulArchivePath(eventId: string): string {
  return `${SOUL_ARCHIVE_PATH_PREFIX}${eventId}`;
}

export function parseSoulMemoryPath(input: string): string | null {
  const normalized = input.trim().replace(/^\/+/, "");
  if (!normalized.toLowerCase().startsWith(SOUL_MEMORY_PATH_PREFIX)) {
    return null;
  }
  const itemId = normalized.slice(SOUL_MEMORY_PATH_PREFIX.length).trim();
  if (!/^mem_[a-z0-9]+$/i.test(itemId)) {
    return null;
  }
  return itemId;
}

export function parseSoulArchivePath(input: string): string | null {
  const normalized = input.trim().replace(/^\/+/, "");
  if (!normalized.toLowerCase().startsWith(SOUL_ARCHIVE_PATH_PREFIX)) {
    return null;
  }
  const eventId = normalized.slice(SOUL_ARCHIVE_PATH_PREFIX.length).trim();
  if (!/^arch_[a-z0-9]+$/i.test(eventId)) {
    return null;
  }
  return eventId;
}

export function resolveSoulMemoryDbPath(agentId: string): string {
  const stateDir = resolveStateDir(process.env, os.homedir);
  return path.join(stateDir, "memory", "soul", `${normalizeScopeValue(agentId)}.sqlite`);
}

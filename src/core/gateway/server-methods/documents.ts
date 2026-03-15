import crypto from "node:crypto";
import type { Dirent } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { listAgentIds, resolveAgentWorkspaceDir } from "../../../agents/agent-scope.js";
import { isP0FileName } from "../../../agents/p0.js";
import { normalizeAgentId } from "../../../routing/session-key.js";
import { loadConfig } from "../../config/config.js";
import {
  ErrorCodes,
  errorShape,
  formatValidationErrors,
  validateDocumentsListParams,
  validateDocumentsReadParams,
} from "../protocol/index.js";
import type { GatewayRequestHandlers } from "./types.js";

const DOCUMENT_EXTENSIONS = new Set([
  ".md",
  ".mdx",
  ".txt",
  ".json",
  ".jsonl",
  ".yaml",
  ".yml",
  ".csv",
]);
const IGNORED_DIRS = new Set([
  ".git",
  ".marv",
  ".next",
  ".turbo",
  ".cache",
  "coverage",
  "dist",
  "build",
  "node_modules",
]);
const MAX_FILES_PER_ROOT = 300;
const MAX_SCAN_DEPTH = 6;
const DEFAULT_DOCUMENT_LIMIT = 200;
const DEFAULT_PREVIEW_BYTES = 320;
const DEFAULT_READ_MAX_BYTES = 120_000;

type WorkspaceRoot = {
  id: string;
  agentId: string;
  agentIds: string[];
  label: string;
  path: string;
};

type DocumentEntry = {
  rootId: string;
  agentId: string;
  agentIds: string[];
  relativePath: string;
  name: string;
  category: string;
  extension: string;
  size: number;
  mtimeMs: number;
  preview?: string;
};

function makeRootId(rootPath: string): string {
  return crypto.createHash("sha1").update(rootPath).digest("hex").slice(0, 12);
}

function buildWorkspaceRoots(params: { agentId?: string }): WorkspaceRoot[] {
  const cfg = loadConfig();
  const knownAgents = listAgentIds(cfg);
  if (params.agentId && !knownAgents.includes(params.agentId)) {
    return [];
  }
  const agentIds = params.agentId ? [params.agentId] : knownAgents;
  const byPath = new Map<string, WorkspaceRoot>();
  for (const agentId of agentIds) {
    const workspacePath = resolveAgentWorkspaceDir(cfg, agentId);
    const existing = byPath.get(workspacePath);
    if (existing) {
      if (!existing.agentIds.includes(agentId)) {
        existing.agentIds.push(agentId);
      }
      continue;
    }
    byPath.set(workspacePath, {
      id: makeRootId(workspacePath),
      agentId,
      agentIds: [agentId],
      label: path.basename(workspacePath) || agentId,
      path: workspacePath,
    });
  }
  return [...byPath.values()].map((root) => ({
    ...root,
    agentIds: root.agentIds.toSorted(),
  }));
}

function isTextDocument(filePath: string) {
  return DOCUMENT_EXTENSIONS.has(path.extname(filePath).toLowerCase());
}

function categoryForPath(relativePath: string): string {
  const firstSegment = relativePath.split(path.sep).filter(Boolean)[0];
  return firstSegment || "root";
}

function normalizeRelativePath(rootPath: string, candidate: string): string | null {
  const resolvedRoot = path.resolve(rootPath);
  const resolvedPath = path.resolve(resolvedRoot, candidate);
  const relativePath = path.relative(resolvedRoot, resolvedPath);
  if (!relativePath || relativePath.startsWith("..") || path.isAbsolute(relativePath)) {
    return null;
  }
  return relativePath;
}

async function readPreview(filePath: string): Promise<string | undefined> {
  try {
    const handle = await fs.open(filePath, "r");
    try {
      const buffer = Buffer.alloc(DEFAULT_PREVIEW_BYTES);
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
      if (bytesRead <= 0) {
        return undefined;
      }
      return buffer
        .subarray(0, bytesRead)
        .toString("utf-8")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 160);
    } finally {
      await handle.close();
    }
  } catch {
    return undefined;
  }
}

async function scanWorkspaceRoot(root: WorkspaceRoot): Promise<DocumentEntry[]> {
  const entries: DocumentEntry[] = [];

  async function walk(currentDir: string, depth: number) {
    if (depth > MAX_SCAN_DEPTH || entries.length >= MAX_FILES_PER_ROOT) {
      return;
    }
    let dirents: Dirent[];
    try {
      dirents = await fs.readdir(currentDir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const dirent of dirents) {
      if (entries.length >= MAX_FILES_PER_ROOT) {
        return;
      }
      const name = dirent.name;
      if (!name || name === "." || name === "..") {
        continue;
      }
      if (dirent.isDirectory()) {
        if (IGNORED_DIRS.has(name) || name.startsWith(".")) {
          continue;
        }
        await walk(path.join(currentDir, name), depth + 1);
        continue;
      }
      if (!dirent.isFile()) {
        continue;
      }
      const filePath = path.join(currentDir, name);
      if (isP0FileName(name)) {
        continue;
      }
      if (!isTextDocument(filePath)) {
        continue;
      }
      const relativePath = normalizeRelativePath(root.path, filePath);
      if (!relativePath) {
        continue;
      }
      let stat: Awaited<ReturnType<typeof fs.stat>>;
      try {
        stat = await fs.stat(filePath);
      } catch {
        continue;
      }
      entries.push({
        rootId: root.id,
        agentId: root.agentId,
        agentIds: root.agentIds,
        relativePath: relativePath.split(path.sep).join("/"),
        name,
        category: categoryForPath(relativePath),
        extension: path.extname(name).toLowerCase(),
        size: stat.size,
        mtimeMs: stat.mtimeMs,
        preview: await readPreview(filePath),
      });
    }
  }

  await walk(root.path, 0);
  return entries;
}

export const documentsHandlers: GatewayRequestHandlers = {
  "documents.list": async ({ params, respond }) => {
    if (!validateDocumentsListParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid documents.list params: ${formatValidationErrors(validateDocumentsListParams.errors)}`,
        ),
      );
      return;
    }
    const agentIdRaw = typeof params.agentId === "string" ? params.agentId.trim() : "";
    const agentId = agentIdRaw ? normalizeAgentId(agentIdRaw) : undefined;
    const roots = buildWorkspaceRoots({ agentId });
    if (agentId && roots.length === 0) {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.INVALID_REQUEST, `unknown agent id "${agentIdRaw}"`),
      );
      return;
    }
    const rootId = typeof params.rootId === "string" ? params.rootId.trim() : "";
    const selectedRoots = rootId ? roots.filter((root) => root.id === rootId) : roots;
    if (rootId && selectedRoots.length === 0) {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.INVALID_REQUEST, `unknown document root "${rootId}"`),
      );
      return;
    }
    const scanned = await Promise.all(
      selectedRoots.map(async (root) => [root, await scanWorkspaceRoot(root)] as const),
    );
    const query =
      typeof params.query === "string" && params.query.trim()
        ? params.query.trim().toLowerCase()
        : "";
    const limit = typeof params.limit === "number" ? params.limit : DEFAULT_DOCUMENT_LIMIT;
    const sort = params.sort === "path" ? "path" : "recent";
    const rootsWithCounts = scanned.map(([root, entries]) => ({
      ...root,
      fileCount: entries.length,
    }));
    let items = scanned.flatMap(([, entries]) => entries);
    if (query) {
      items = items.filter((entry) => {
        const haystack =
          `${entry.relativePath}\n${entry.name}\n${entry.category}\n${entry.preview ?? ""}`.toLowerCase();
        return haystack.includes(query);
      });
    }
    items =
      sort === "path"
        ? items.toSorted((a, b) =>
            a.relativePath.localeCompare(b.relativePath, undefined, { sensitivity: "base" }),
          )
        : items.toSorted((a, b) => b.mtimeMs - a.mtimeMs);
    respond(
      true,
      {
        updatedAt: Date.now(),
        roots: rootsWithCounts,
        items: items.slice(0, limit),
      },
      undefined,
    );
  },
  "documents.read": async ({ params, respond }) => {
    if (!validateDocumentsReadParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid documents.read params: ${formatValidationErrors(validateDocumentsReadParams.errors)}`,
        ),
      );
      return;
    }
    const roots = buildWorkspaceRoots({});
    const root = roots.find((entry) => entry.id === params.rootId);
    if (!root) {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.INVALID_REQUEST, `unknown document root "${params.rootId}"`),
      );
      return;
    }
    const relativePath = normalizeRelativePath(root.path, String(params.relativePath));
    if (!relativePath) {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.INVALID_REQUEST, "relativePath must stay within the workspace root"),
      );
      return;
    }
    const absolutePath = path.join(root.path, relativePath);
    if (!isTextDocument(absolutePath)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `unsupported document type "${path.extname(absolutePath).toLowerCase() || "unknown"}"`,
        ),
      );
      return;
    }
    try {
      const stat = await fs.stat(absolutePath);
      const maxBytes =
        typeof params.maxBytes === "number" ? params.maxBytes : DEFAULT_READ_MAX_BYTES;
      const contentBuffer = await fs.readFile(absolutePath);
      const truncated = contentBuffer.byteLength > maxBytes;
      const slice = truncated ? contentBuffer.subarray(0, maxBytes) : contentBuffer;
      respond(
        true,
        {
          rootId: root.id,
          agentId: root.agentId,
          agentIds: root.agentIds,
          relativePath: relativePath.split(path.sep).join("/"),
          name: path.basename(relativePath),
          size: stat.size,
          mtimeMs: stat.mtimeMs,
          content: slice.toString("utf-8"),
          truncated,
        },
        undefined,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      respond(false, undefined, errorShape(ErrorCodes.UNAVAILABLE, message));
    }
  },
};

import fs from "node:fs/promises";
import path from "node:path";
import { listAgentIds, resolveAgentWorkspaceDir } from "../../../agents/agent-scope.js";
import {
  isP0FileName,
  projectAgentP0FilesFromConfig,
  readAgentP0FilesFromWorkspace,
  setAgentP0Sections,
} from "../../../agents/p0.js";
import {
  DEFAULT_AGENTS_FILENAME,
  DEFAULT_BOOTSTRAP_FILENAME,
  DEFAULT_HEARTBEAT_FILENAME,
  DEFAULT_IDENTITY_FILENAME,
  DEFAULT_MEMORY_ALT_FILENAME,
  DEFAULT_MEMORY_FILENAME,
  DEFAULT_SOUL_FILENAME,
  DEFAULT_TOOLS_FILENAME,
  DEFAULT_USER_FILENAME,
  isWorkspaceOnboardingCompleted,
} from "../../../agents/workspace.js";
import { normalizeAgentId } from "../../../routing/session-key.js";
import {
  loadConfig,
  readConfigFileSnapshotForWrite,
  writeConfigFile,
} from "../../config/config.js";
import {
  ErrorCodes,
  errorShape,
  formatValidationErrors,
  validateAgentsCreateParams,
  validateAgentsDeleteParams,
  validateAgentsFilesGetParams,
  validateAgentsFilesListParams,
  validateAgentsFilesSetParams,
  validateAgentsListParams,
  validateAgentsUpdateParams,
} from "../protocol/index.js";
import { listAgentsForGateway } from "../session-utils.js";
import type { GatewayRequestHandlers, RespondFn } from "./types.js";

const BOOTSTRAP_FILE_NAMES = [
  DEFAULT_AGENTS_FILENAME,
  DEFAULT_SOUL_FILENAME,
  DEFAULT_TOOLS_FILENAME,
  DEFAULT_IDENTITY_FILENAME,
  DEFAULT_USER_FILENAME,
  DEFAULT_HEARTBEAT_FILENAME,
  DEFAULT_BOOTSTRAP_FILENAME,
] as const;
const BOOTSTRAP_FILE_NAMES_POST_ONBOARDING = BOOTSTRAP_FILE_NAMES.filter(
  (name) => name !== DEFAULT_BOOTSTRAP_FILENAME,
);

const MEMORY_FILE_NAMES = [DEFAULT_MEMORY_FILENAME, DEFAULT_MEMORY_ALT_FILENAME] as const;

const ALLOWED_FILE_NAMES = new Set<string>([...BOOTSTRAP_FILE_NAMES, ...MEMORY_FILE_NAMES]);

function resolveAgentWorkspaceFileOrRespondError(
  params: Record<string, unknown>,
  respond: RespondFn,
): {
  cfg: ReturnType<typeof loadConfig>;
  agentId: string;
  workspaceDir: string;
  name: string;
} | null {
  const cfg = loadConfig();
  const rawAgentId = params.agentId;
  const agentId = resolveAgentIdOrError(
    typeof rawAgentId === "string" || typeof rawAgentId === "number" ? String(rawAgentId) : "",
    cfg,
  );
  if (!agentId) {
    respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "unknown agent id"));
    return null;
  }
  const rawName = params.name;
  const name = (
    typeof rawName === "string" || typeof rawName === "number" ? String(rawName) : ""
  ).trim();
  if (!ALLOWED_FILE_NAMES.has(name)) {
    respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, `unsupported file "${name}"`));
    return null;
  }
  const workspaceDir = resolveAgentWorkspaceDir(cfg, agentId);
  return { cfg, agentId, workspaceDir, name };
}

type FileMeta = {
  size: number;
  updatedAtMs: number;
};

async function statFile(filePath: string): Promise<FileMeta | null> {
  try {
    const stat = await fs.stat(filePath);
    if (!stat.isFile()) {
      return null;
    }
    return {
      size: stat.size,
      updatedAtMs: Math.floor(stat.mtimeMs),
    };
  } catch {
    return null;
  }
}

async function listAgentFiles(workspaceDir: string, options?: { hideBootstrap?: boolean }) {
  const files: Array<{
    name: string;
    path: string;
    missing: boolean;
    size?: number;
    updatedAtMs?: number;
  }> = [];

  const bootstrapFileNames = options?.hideBootstrap
    ? BOOTSTRAP_FILE_NAMES_POST_ONBOARDING
    : BOOTSTRAP_FILE_NAMES;
  for (const name of bootstrapFileNames) {
    const filePath = path.join(workspaceDir, name);
    const meta = await statFile(filePath);
    if (meta) {
      files.push({
        name,
        path: filePath,
        missing: false,
        size: meta.size,
        updatedAtMs: meta.updatedAtMs,
      });
    } else {
      files.push({ name, path: filePath, missing: true });
    }
  }

  const primaryMemoryPath = path.join(workspaceDir, DEFAULT_MEMORY_FILENAME);
  const primaryMeta = await statFile(primaryMemoryPath);
  if (primaryMeta) {
    files.push({
      name: DEFAULT_MEMORY_FILENAME,
      path: primaryMemoryPath,
      missing: false,
      size: primaryMeta.size,
      updatedAtMs: primaryMeta.updatedAtMs,
    });
  } else {
    const altMemoryPath = path.join(workspaceDir, DEFAULT_MEMORY_ALT_FILENAME);
    const altMeta = await statFile(altMemoryPath);
    if (altMeta) {
      files.push({
        name: DEFAULT_MEMORY_ALT_FILENAME,
        path: altMemoryPath,
        missing: false,
        size: altMeta.size,
        updatedAtMs: altMeta.updatedAtMs,
      });
    } else {
      files.push({ name: DEFAULT_MEMORY_FILENAME, path: primaryMemoryPath, missing: true });
    }
  }

  return files;
}

function resolveAgentIdOrError(agentIdRaw: string, cfg: ReturnType<typeof loadConfig>) {
  const agentId = normalizeAgentId(agentIdRaw);
  const allowed = new Set(listAgentIds(cfg));
  if (!allowed.has(agentId)) {
    return null;
  }
  return agentId;
}

function respondLegacyTopLevelAgentsRemoved(respond: RespondFn) {
  respond(
    false,
    undefined,
    errorShape(
      ErrorCodes.INVALID_REQUEST,
      'Top-level agent create/update/delete was removed. Marv now keeps a single durable "main" agent and uses enhanced subagents for delegation.',
    ),
  );
}

export const agentsHandlers: GatewayRequestHandlers = {
  "agents.list": ({ params, respond }) => {
    if (!validateAgentsListParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid agents.list params: ${formatValidationErrors(validateAgentsListParams.errors)}`,
        ),
      );
      return;
    }

    const cfg = loadConfig();
    const result = listAgentsForGateway(cfg);
    respond(true, result, undefined);
  },
  "agents.create": async ({ params, respond }) => {
    if (!validateAgentsCreateParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid agents.create params: ${formatValidationErrors(
            validateAgentsCreateParams.errors,
          )}`,
        ),
      );
      return;
    }
    respondLegacyTopLevelAgentsRemoved(respond);
  },
  "agents.update": async ({ params, respond }) => {
    if (!validateAgentsUpdateParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid agents.update params: ${formatValidationErrors(
            validateAgentsUpdateParams.errors,
          )}`,
        ),
      );
      return;
    }
    respondLegacyTopLevelAgentsRemoved(respond);
  },
  "agents.delete": async ({ params, respond }) => {
    if (!validateAgentsDeleteParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid agents.delete params: ${formatValidationErrors(
            validateAgentsDeleteParams.errors,
          )}`,
        ),
      );
      return;
    }
    respondLegacyTopLevelAgentsRemoved(respond);
  },
  "agents.files.list": async ({ params, respond }) => {
    if (!validateAgentsFilesListParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid agents.files.list params: ${formatValidationErrors(
            validateAgentsFilesListParams.errors,
          )}`,
        ),
      );
      return;
    }
    const cfg = loadConfig();
    const agentId = resolveAgentIdOrError(String(params.agentId ?? ""), cfg);
    if (!agentId) {
      respond(false, undefined, errorShape(ErrorCodes.INVALID_REQUEST, "unknown agent id"));
      return;
    }
    const workspaceDir = resolveAgentWorkspaceDir(cfg, agentId);
    await projectAgentP0FilesFromConfig(cfg);
    let hideBootstrap = false;
    try {
      hideBootstrap = await isWorkspaceOnboardingCompleted(workspaceDir);
    } catch {
      // Fall back to showing BOOTSTRAP if workspace state cannot be read.
    }
    const files = await listAgentFiles(workspaceDir, { hideBootstrap });
    respond(true, { agentId, workspace: workspaceDir, files }, undefined);
  },
  "agents.files.get": async ({ params, respond }) => {
    if (!validateAgentsFilesGetParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid agents.files.get params: ${formatValidationErrors(
            validateAgentsFilesGetParams.errors,
          )}`,
        ),
      );
      return;
    }
    const resolved = resolveAgentWorkspaceFileOrRespondError(params, respond);
    if (!resolved) {
      return;
    }
    const { cfg, agentId, workspaceDir, name } = resolved;
    await projectAgentP0FilesFromConfig(cfg);
    const filePath = path.join(workspaceDir, name);
    const meta = await statFile(filePath);
    if (!meta) {
      respond(
        true,
        {
          agentId,
          workspace: workspaceDir,
          file: { name, path: filePath, missing: true },
        },
        undefined,
      );
      return;
    }
    const content = await fs.readFile(filePath, "utf-8");
    respond(
      true,
      {
        agentId,
        workspace: workspaceDir,
        file: {
          name,
          path: filePath,
          missing: false,
          size: meta.size,
          updatedAtMs: meta.updatedAtMs,
          content,
        },
      },
      undefined,
    );
  },
  "agents.files.set": async ({ params, respond }) => {
    if (!validateAgentsFilesSetParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid agents.files.set params: ${formatValidationErrors(
            validateAgentsFilesSetParams.errors,
          )}`,
        ),
      );
      return;
    }
    const resolved = resolveAgentWorkspaceFileOrRespondError(params, respond);
    if (!resolved) {
      return;
    }
    const { cfg, agentId, workspaceDir, name } = resolved;
    await fs.mkdir(workspaceDir, { recursive: true });
    const content = String(params.content ?? "");
    if (isP0FileName(name)) {
      const { snapshot, writeOptions } = await readConfigFileSnapshotForWrite();
      const baseConfig = snapshot.valid ? snapshot.config : cfg;
      const imported = await readAgentP0FilesFromWorkspace(workspaceDir);
      const section =
        name === DEFAULT_SOUL_FILENAME
          ? "soul"
          : name === DEFAULT_IDENTITY_FILENAME
            ? "identity"
            : "user";
      const nextConfig = setAgentP0Sections(baseConfig, {
        ...imported,
        [section]: content,
      });
      await writeConfigFile(nextConfig, writeOptions);
    } else {
      const filePath = path.join(workspaceDir, name);
      await fs.writeFile(filePath, content, "utf-8");
    }
    const filePath = path.join(workspaceDir, name);
    const meta = await statFile(filePath);
    respond(
      true,
      {
        ok: true,
        agentId,
        workspace: workspaceDir,
        file: {
          name,
          path: filePath,
          missing: false,
          size: meta?.size,
          updatedAtMs: meta?.updatedAtMs,
          content,
        },
      },
      undefined,
    );
  },
};

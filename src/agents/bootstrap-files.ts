import type { MarvConfig } from "../core/config/config.js";
import { applyBootstrapHookOverrides } from "./bootstrap-hooks.js";
import type { EmbeddedContextFile } from "./runner/pi-embedded-helpers.js";
import {
  buildBootstrapContextFiles,
  resolveBootstrapMaxChars,
  resolveBootstrapTotalMaxChars,
  trimToAnchor,
} from "./runner/pi-embedded-helpers.js";
import {
  DEFAULT_AGENTS_FILENAME,
  DEFAULT_SOUL_FILENAME,
  DEFAULT_TOOLS_FILENAME,
  filterBootstrapFilesForSession,
  loadWorkspaceBootstrapFiles,
  type WorkspaceBootstrapFile,
} from "./workspace.js";

function isAutoRecallEnabled(config?: MarvConfig): boolean {
  return config?.memory?.autoRecall?.enabled !== false;
}

function filterBootstrapFilesForAutoRecall(
  files: WorkspaceBootstrapFile[],
  params: { autoRecallEnabled: boolean; sessionKey?: string; sessionId?: string },
): WorkspaceBootstrapFile[] {
  if (!params.autoRecallEnabled) {
    return files;
  }
  const sessionLabel = params.sessionKey ?? params.sessionId ?? "";
  const keep = new Set([DEFAULT_SOUL_FILENAME, DEFAULT_TOOLS_FILENAME]);
  if (sessionLabel.includes("subagent:") || sessionLabel.includes("cron:")) {
    keep.add(DEFAULT_AGENTS_FILENAME);
  }
  return files.filter((file) => keep.has(file.name));
}

function applyAutoRecallBootstrapTransforms(
  files: WorkspaceBootstrapFile[],
  autoRecallEnabled: boolean,
): WorkspaceBootstrapFile[] {
  if (!autoRecallEnabled) {
    return files;
  }
  return files.map((file) => {
    if (file.name !== DEFAULT_SOUL_FILENAME || file.missing || !file.content) {
      return file;
    }
    return {
      ...file,
      content: trimToAnchor(file.content),
    };
  });
}

export function makeBootstrapWarn(params: {
  sessionLabel: string;
  warn?: (message: string) => void;
}): ((message: string) => void) | undefined {
  if (!params.warn) {
    return undefined;
  }
  return (message: string) => params.warn?.(`${message} (sessionKey=${params.sessionLabel})`);
}

export async function resolveBootstrapFilesForRun(params: {
  workspaceDir: string;
  config?: MarvConfig;
  sessionKey?: string;
  sessionId?: string;
  agentId?: string;
}): Promise<WorkspaceBootstrapFile[]> {
  const sessionKey = params.sessionKey ?? params.sessionId;
  const bootstrapFiles = filterBootstrapFilesForSession(
    await loadWorkspaceBootstrapFiles(params.workspaceDir),
    sessionKey,
  );

  return applyBootstrapHookOverrides({
    files: bootstrapFiles,
    workspaceDir: params.workspaceDir,
    config: params.config,
    sessionKey: params.sessionKey,
    sessionId: params.sessionId,
    agentId: params.agentId,
  });
}

export async function resolveBootstrapContextForRun(params: {
  workspaceDir: string;
  config?: MarvConfig;
  sessionKey?: string;
  sessionId?: string;
  agentId?: string;
  warn?: (message: string) => void;
}): Promise<{
  bootstrapFiles: WorkspaceBootstrapFile[];
  contextFiles: EmbeddedContextFile[];
}> {
  const bootstrapFiles = await resolveBootstrapFilesForRun(params);
  const autoRecallEnabled = isAutoRecallEnabled(params.config);
  const filteredBootstrapFiles = filterBootstrapFilesForAutoRecall(bootstrapFiles, {
    autoRecallEnabled,
    sessionKey: params.sessionKey,
    sessionId: params.sessionId,
  });
  const transformedBootstrapFiles = applyAutoRecallBootstrapTransforms(
    filteredBootstrapFiles,
    autoRecallEnabled,
  );
  const contextFiles = buildBootstrapContextFiles(transformedBootstrapFiles, {
    maxChars: resolveBootstrapMaxChars(params.config),
    totalMaxChars: resolveBootstrapTotalMaxChars(params.config),
    warn: params.warn,
  });
  return { bootstrapFiles, contextFiles };
}

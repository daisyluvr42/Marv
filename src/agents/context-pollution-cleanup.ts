import path from "node:path";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { SessionManager } from "@mariozechner/pi-coding-agent";
import type { MarvConfig } from "../core/config/config.js";
import {
  loadSessionStore,
  resolveSessionFilePath,
  resolveStorePath,
  type SessionEntry,
} from "../core/config/sessions.js";
import { emitSessionTranscriptUpdate } from "../core/session/transcript-events.js";
import {
  listTaskContextEntries,
  removeTaskContextEntries,
  resolveTaskContextDbPath,
} from "../memory/task-context/index.js";
import { resolveAgentIdFromSessionKey } from "../routing/session-key.js";
import {
  scanContextPollution,
  taskContextTurnsFromEntries,
  transcriptTurnsFromSessionEntries,
  type ContextPollutionScan,
  type ContextPollutionViolation,
  type ReplyFormatPreferences,
} from "./context-pollution.js";
import { acquireSessionWriteLock } from "./session-write-lock.js";

type SessionTranscriptEntry = {
  id?: string;
  parentId?: string | null;
  type?: string;
  message?: AgentMessage;
};

type SessionManagerRewriteHandle = {
  fileEntries: SessionTranscriptEntry[];
  leafId?: string | null;
  _buildIndex: () => void;
  _rewriteFile: () => void;
};

export type ContextPollutionInspection = {
  sessionKey: string;
  agentId: string;
  sessionFile?: string;
  taskContextPath?: string;
  preferences: ReplyFormatPreferences;
  transcript: {
    violations: ContextPollutionViolation[];
    removableIds: string[];
    sanitizedIds: string[];
  };
  taskContext: {
    violations: ContextPollutionViolation[];
    removableIds: string[];
  };
};

export type ContextPollutionCleanupResult = ContextPollutionInspection & {
  cleaned: {
    transcriptRemoved: number;
    taskContextRemoved: number;
  };
};

function resolveSessionEntry(params: { cfg: MarvConfig; sessionKey: string }): {
  agentId: string;
  storePath: string;
  entry?: SessionEntry;
  sessionFile?: string;
} {
  const agentId = resolveAgentIdFromSessionKey(params.sessionKey);
  const storePath = resolveStorePath(params.cfg.session?.store, { agentId });
  const store = loadSessionStore(storePath, { skipCache: true });
  const entry = store[params.sessionKey];
  if (!entry?.sessionId) {
    return { agentId, storePath };
  }
  const sessionFile = resolveSessionFilePath(entry.sessionId, entry, {
    agentId,
    sessionsDir: path.dirname(storePath),
  });
  return { agentId, storePath, entry, sessionFile };
}

function resolveTranscriptRemovableIds(
  entries: SessionTranscriptEntry[],
  scan: ContextPollutionScan,
): string[] {
  const turns = transcriptTurnsFromSessionEntries(entries);
  const removable = new Set(turns.filter((turn) => turn.removable).map((turn) => turn.id));
  return scan.violations
    .filter((violation) => violation.source === "transcript" && removable.has(violation.id))
    .map((violation) => violation.id);
}

function resolveTranscriptSanitizedIds(
  entries: SessionTranscriptEntry[],
  scan: ContextPollutionScan,
): string[] {
  const turns = transcriptTurnsFromSessionEntries(entries);
  const removable = new Set(turns.filter((turn) => turn.removable).map((turn) => turn.id));
  return scan.violations
    .filter((violation) => violation.source === "transcript" && !removable.has(violation.id))
    .map((violation) => violation.id);
}

function rewriteTranscriptEntries(params: {
  sessionManager: SessionManager;
  removedIds: Set<string>;
  sanitizedIds: Set<string>;
}): { removedCount: number; sanitizedCount: number } {
  const manager = params.sessionManager as unknown as SessionManagerRewriteHandle;
  const currentEntries = Array.isArray(manager.fileEntries) ? manager.fileEntries : [];
  if (currentEntries.length === 0) {
    return { removedCount: 0, sanitizedCount: 0 };
  }

  const byId = new Map(
    currentEntries
      .filter(
        (entry): entry is SessionTranscriptEntry & { id: string } => typeof entry.id === "string",
      )
      .map((entry) => [entry.id, entry]),
  );
  const keptEntries = currentEntries.filter(
    (entry) => !(entry.type === "message" && entry.id && params.removedIds.has(entry.id)),
  );
  const keptIds = new Set(
    keptEntries
      .map((entry) => (typeof entry.id === "string" ? entry.id : undefined))
      .filter((value): value is string => Boolean(value)),
  );
  const resolveParentId = (parentId: string | null | undefined): string | null => {
    let next = parentId ?? null;
    while (next && !keptIds.has(next)) {
      next = byId.get(next)?.parentId ?? null;
    }
    return next;
  };

  let sanitizedCount = 0;
  const rewrittenEntries = keptEntries.map((entry) => {
    if (
      entry.type === "message" &&
      entry.id &&
      params.sanitizedIds.has(entry.id) &&
      entry.message?.role === "assistant" &&
      Array.isArray(entry.message.content)
    ) {
      const nextContent = entry.message.content.filter((block) => block?.type !== "text");
      if (nextContent.length !== entry.message.content.length) {
        sanitizedCount += 1;
        return {
          ...entry,
          message: {
            ...entry.message,
            content: nextContent,
          },
        };
      }
    }
    if (!("parentId" in entry)) {
      return entry;
    }
    const nextParentId = resolveParentId(entry.parentId ?? null);
    if (nextParentId === (entry.parentId ?? null)) {
      return entry;
    }
    return { ...entry, parentId: nextParentId };
  });
  const nextLeafId = resolveParentId(manager.leafId ?? null);
  const removedCount = currentEntries.length - keptEntries.length;

  if (removedCount === 0 && sanitizedCount === 0) {
    return { removedCount: 0, sanitizedCount: 0 };
  }
  manager.fileEntries = rewrittenEntries;
  manager.leafId = nextLeafId;
  manager._buildIndex();
  manager._rewriteFile();
  return { removedCount, sanitizedCount };
}

export function inspectContextPollution(params: {
  cfg: MarvConfig;
  sessionKey: string;
}): ContextPollutionInspection {
  const resolved = resolveSessionEntry(params);
  let transcriptEntries: SessionTranscriptEntry[] = [];
  if (resolved.sessionFile) {
    const sessionManager = SessionManager.open(resolved.sessionFile);
    transcriptEntries = sessionManager.getEntries() as SessionTranscriptEntry[];
  }
  const transcriptTurns = transcriptTurnsFromSessionEntries(transcriptEntries);
  const transcriptScan = scanContextPollution(transcriptTurns);
  const transcriptRemovableIds = resolveTranscriptRemovableIds(transcriptEntries, transcriptScan);
  const transcriptSanitizedIds = resolveTranscriptSanitizedIds(transcriptEntries, transcriptScan);

  const taskEntries = listTaskContextEntries({
    agentId: resolved.agentId,
    taskId: params.sessionKey,
    limit: 5000,
  });
  const taskTurns = taskContextTurnsFromEntries(taskEntries);
  const taskScan = scanContextPollution(taskTurns);
  const taskRemovableIds = taskScan.violations
    .filter((violation) => violation.source === "task-context")
    .map((violation) => violation.id);

  return {
    sessionKey: params.sessionKey,
    agentId: resolved.agentId,
    sessionFile: resolved.sessionFile,
    taskContextPath: resolveTaskContextDbPath({
      agentId: resolved.agentId,
      taskId: params.sessionKey,
    }),
    preferences: {
      noPinyinChinese:
        transcriptScan.preferences.noPinyinChinese || taskScan.preferences.noPinyinChinese,
      noInlineEnglishChinese:
        transcriptScan.preferences.noInlineEnglishChinese ||
        taskScan.preferences.noInlineEnglishChinese,
    },
    transcript: {
      violations: transcriptScan.violations.filter(
        (violation) => violation.source === "transcript",
      ),
      removableIds: transcriptRemovableIds,
      sanitizedIds: transcriptSanitizedIds,
    },
    taskContext: {
      violations: taskScan.violations.filter((violation) => violation.source === "task-context"),
      removableIds: taskRemovableIds,
    },
  };
}

export async function cleanupContextPollution(params: {
  cfg: MarvConfig;
  sessionKey: string;
}): Promise<ContextPollutionCleanupResult> {
  const inspection = inspectContextPollution(params);
  let transcriptRemoved = 0;
  let transcriptSanitized = 0;
  if (
    inspection.sessionFile &&
    (inspection.transcript.removableIds.length > 0 || inspection.transcript.sanitizedIds.length > 0)
  ) {
    const lock = await acquireSessionWriteLock({
      sessionFile: inspection.sessionFile,
      timeoutMs: 10_000,
    });
    try {
      const sessionManager = SessionManager.open(inspection.sessionFile);
      const rewritten = rewriteTranscriptEntries({
        sessionManager,
        removedIds: new Set(inspection.transcript.removableIds),
        sanitizedIds: new Set(inspection.transcript.sanitizedIds),
      });
      transcriptRemoved = rewritten.removedCount;
      transcriptSanitized = rewritten.sanitizedCount;
      if (transcriptRemoved > 0 || transcriptSanitized > 0) {
        emitSessionTranscriptUpdate(inspection.sessionFile);
      }
    } finally {
      await lock.release();
    }
  }

  const taskCleanup = removeTaskContextEntries({
    agentId: inspection.agentId,
    taskId: inspection.sessionKey,
    entryIds: inspection.taskContext.removableIds,
  });

  return {
    ...inspection,
    cleaned: {
      transcriptRemoved: transcriptRemoved + transcriptSanitized,
      taskContextRemoved: taskCleanup.removedCount,
    },
  };
}

export function summarizeContextPollution(inspection: ContextPollutionInspection): string {
  const parts = [];
  if (!inspection.preferences.noPinyinChinese && !inspection.preferences.noInlineEnglishChinese) {
    parts.push("No explicit reply-format pollution preference is currently detected.");
  } else {
    const enabled = [];
    if (inspection.preferences.noPinyinChinese) {
      enabled.push("no automatic pinyin");
    }
    if (inspection.preferences.noInlineEnglishChinese) {
      enabled.push("no unnecessary inline English");
    }
    parts.push(`Active reply-format preferences: ${enabled.join(", ")}.`);
  }

  const transcriptCount = inspection.transcript.removableIds.length;
  const transcriptSanitizedCount = inspection.transcript.sanitizedIds.length;
  const taskCount = inspection.taskContext.removableIds.length;
  if (transcriptCount === 0 && transcriptSanitizedCount === 0 && taskCount === 0) {
    parts.push("No removable context pollution is currently detected.");
  } else {
    parts.push(
      `Removable pollution detected: transcript ${transcriptCount + transcriptSanitizedCount}, task context ${taskCount}.`,
    );
  }
  return parts.join(" ");
}

import { Type } from "@sinclair/typebox";
import { loadConfig, type MarvConfig } from "../../core/config/config.js";
import {
  loadSessionStore,
  resolveStorePath,
  type SessionEntry,
} from "../../core/config/sessions.js";
import { resolveSessionModelRef } from "../../core/gateway/session-utils.js";
import { resolveAgentIdFromSessionKey } from "../../routing/session-key.js";
import { resolveAgentDir } from "../agent-scope.js";
import {
  cleanupContextPollution,
  inspectContextPollution,
  summarizeContextPollution,
} from "../context-pollution-cleanup.js";
import { resolveRuntimeModelPlan } from "../model/model-pool.js";
import { resolveDefaultModelForAgent } from "../model/model-selection.js";
import {
  readRuntimeModelRegistry,
  resolveRuntimeRegistryPathForDisplay,
} from "../model/runtime-model-registry.js";
import type { AnyAgentTool } from "./common.js";
import { ToolInputError } from "./common.js";
import { callGatewayTool } from "./gateway.js";
import { createSessionStatusTool } from "./session-status-tool.js";

const SelfInspectingToolSchema = Type.Object(
  {
    query: Type.Optional(Type.String()),
    cleanupContextPollution: Type.Optional(Type.Boolean()),
  },
  { additionalProperties: false },
);

type SelfInspectingQuery =
  | "summary"
  | "runtime"
  | "settings"
  | "models"
  | "tasks"
  | "context"
  | "tools"
  | "all";

type CronStatusResult = {
  enabled?: boolean;
  storePath?: string;
  nextWakeAtMs?: number;
};

type CronJobSummary = {
  id?: string;
  name?: string;
  enabled?: boolean;
  schedule?: {
    kind?: string;
    at?: string;
    everyMs?: number;
    expr?: string;
    tz?: string;
  };
  state?: {
    nextRunAtMs?: number;
    lastRunAtMs?: number;
    lastStatus?: string;
  };
};

function normalizeTextQuery(raw: string): string {
  return raw
    .trim()
    .toLowerCase()
    .replace(/[?？!！,，。:：;；/\\]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeQuery(raw: unknown): SelfInspectingQuery {
  const value = typeof raw === "string" ? normalizeTextQuery(raw) : "";
  switch (value) {
    case "runtime":
    case "settings":
    case "models":
    case "tasks":
    case "context":
    case "tools":
    case "all":
      return value;
    default:
      break;
  }

  if (!value) {
    return "summary";
  }

  const hits = new Set<SelfInspectingQuery>();
  const matchesAny = (patterns: string[]) => patterns.some((pattern) => value.includes(pattern));

  if (
    matchesAny([
      "all",
      "full",
      "complete",
      "everything",
      "overview",
      "overall",
      "report",
      "summary report",
      "self check",
      "self-check",
      "自检",
      "完整状态",
      "全部状态",
      "所有状态",
      "总体情况",
      "汇报状态",
      "报告状态",
      "目前的状态",
      "当前状态",
    ])
  ) {
    hits.add("all");
  }
  if (
    matchesAny([
      "runtime",
      "session",
      "status",
      "state",
      "running",
      "当前运行",
      "运行时",
      "会话",
      "连接状态",
      "状态",
    ])
  ) {
    hits.add("runtime");
  }
  if (
    matchesAny([
      "setting",
      "settings",
      "configuration",
      "config",
      "override",
      "overrides",
      "preference",
      "preferences",
      "设定",
      "设置",
      "配置",
      "偏好",
      "override",
      "当前设置",
      "当前设定",
    ])
  ) {
    hits.add("settings");
  }
  if (
    matchesAny([
      "model",
      "models",
      "pool",
      "registry",
      "provider",
      "providers",
      "model list",
      "available model",
      "available models",
      "模型",
      "模型列表",
      "可用模型",
      "模型池",
      "模型注册表",
      "provider 列表",
      "provider列表",
    ])
  ) {
    hits.add("models");
  }
  if (
    matchesAny([
      "task",
      "tasks",
      "cron",
      "schedule",
      "scheduled",
      "scheduler",
      "reminder",
      "reminders",
      "job",
      "jobs",
      "定时任务",
      "任务",
      "计划任务",
      "定时",
      "日程",
      "提醒",
      "cron 任务",
    ])
  ) {
    hits.add("tasks");
  }
  if (
    matchesAny([
      "context",
      "memory",
      "pollution",
      "history",
      "上下文",
      "污染",
      "记忆",
      "历史",
      "上下文污染",
    ])
  ) {
    hits.add("context");
  }
  if (
    matchesAny([
      "tool",
      "tools",
      "capability",
      "capabilities",
      "limit",
      "limits",
      "permission",
      "permissions",
      "工具",
      "能力",
      "限制",
      "权限",
      "可用工具",
    ])
  ) {
    hits.add("tools");
  }

  if (hits.has("all")) {
    return "all";
  }
  if (hits.size > 1) {
    return "all";
  }
  const [firstHit] = hits;
  if (firstHit) {
    return firstHit;
  }
  return "summary";
}

function formatIsoTime(ms: number | undefined): string | undefined {
  if (typeof ms !== "number" || !Number.isFinite(ms) || ms <= 0) {
    return undefined;
  }
  return new Date(ms).toISOString();
}

function formatCronSchedule(job: CronJobSummary): string {
  const schedule = job.schedule;
  if (!schedule?.kind) {
    return "unknown schedule";
  }
  if (schedule.kind === "at") {
    return schedule.at ? `at ${schedule.at}` : "at";
  }
  if (schedule.kind === "every") {
    return schedule.everyMs ? `every ${schedule.everyMs}ms` : "every";
  }
  if (schedule.kind === "cron") {
    return schedule.expr
      ? `cron ${schedule.expr}${schedule.tz ? ` (${schedule.tz})` : ""}`
      : "cron";
  }
  return schedule.kind;
}

function formatSettingsSection(params: {
  sessionKey: string;
  sessionEntry?: SessionEntry;
  directUserInstruction?: boolean;
}): string {
  const entry = params.sessionEntry;
  const lines = [`Session key: ${params.sessionKey}`];
  const pushLine = (label: string, value: unknown) => {
    if (typeof value === "string" && value.trim()) {
      lines.push(`${label}: ${value.trim()}`);
      return;
    }
    if (typeof value === "number" && Number.isFinite(value)) {
      lines.push(`${label}: ${value}`);
      return;
    }
    if (typeof value === "boolean") {
      lines.push(`${label}: ${value ? "true" : "false"}`);
    }
  };

  pushLine("Session id", entry?.sessionId);
  if (entry?.providerOverride || entry?.modelOverride) {
    const provider = entry.providerOverride?.trim();
    const model = entry.modelOverride?.trim();
    lines.push(`Model override: ${provider ? `${provider}/` : ""}${model ?? "(unset)"}`);
  }
  pushLine("Auth profile override", entry?.authProfileOverride);
  pushLine("Thinking level", entry?.thinkingLevel);
  pushLine("Verbose level", entry?.verboseLevel);
  pushLine("Reasoning level", entry?.reasoningLevel);
  pushLine("Response usage", entry?.responseUsage);
  pushLine("Elevated level", entry?.elevatedLevel);
  pushLine("Exec host", entry?.execHost);
  pushLine("Exec security", entry?.execSecurity);
  pushLine("Exec ask", entry?.execAsk);
  pushLine("Exec node", entry?.execNode);
  pushLine("Queue mode", entry?.queueMode);
  pushLine("Queue debounce ms", entry?.queueDebounceMs);
  pushLine("Queue cap", entry?.queueCap);
  pushLine("Queue drop", entry?.queueDrop);
  if (typeof entry?.updatedAt === "number") {
    lines.push(`Session updated at: ${new Date(entry.updatedAt).toISOString()}`);
  }
  if (params.directUserInstruction === false) {
    lines.push("Self-setting changes are currently limited because the instruction is indirect.");
  }
  if (lines.length === 1) {
    lines.push("No session-specific overrides are currently set.");
  }
  return lines.join("\n");
}

function formatModelsSection(params: {
  currentModel: { provider: string; model: string };
  defaultModel: { provider: string; model: string };
  runtimePlan: ReturnType<typeof resolveRuntimeModelPlan>;
  registry: ReturnType<typeof readRuntimeModelRegistry>;
}): string {
  const lines = [
    `Current model: ${params.currentModel.provider}/${params.currentModel.model}`,
    `Default model: ${params.defaultModel.provider}/${params.defaultModel.model}`,
    `Model pool: ${params.runtimePlan.poolName}`,
  ];
  if (params.registry) {
    lines.push(
      `Runtime registry: ${resolveRuntimeRegistryPathForDisplay()} (${params.registry.models.length} models)`,
    );
    if (params.registry.lastSuccessfulRefreshAt) {
      lines.push(
        `Registry last refreshed: ${new Date(params.registry.lastSuccessfulRefreshAt).toISOString()}`,
      );
    }
  }
  const candidateProviders = [
    ...new Set(params.runtimePlan.candidates.map((entry) => entry.provider)),
  ];
  if (candidateProviders.length > 0) {
    lines.push(`Candidate providers: ${candidateProviders.join(", ")}`);
  }
  if (params.runtimePlan.candidates.length > 0) {
    lines.push(`Runnable candidate count: ${params.runtimePlan.candidates.length}`);
    lines.push(
      `Runnable candidates: ${params.runtimePlan.candidates.map((entry) => entry.ref).join(", ")}`,
    );
  }
  const unavailable = params.runtimePlan.configured
    .filter((entry) => !entry.available)
    .map((entry) =>
      entry.availabilityReason ? `${entry.ref} (${entry.availabilityReason})` : entry.ref,
    );
  if (unavailable.length > 0) {
    lines.push(`Unavailable configured models: ${unavailable.join(", ")}`);
  }
  return lines.join("\n");
}

function formatTasksSection(params: {
  cronStatus: CronStatusResult | null;
  cronJobs: CronJobSummary[];
  cronError?: string;
}): string {
  if (params.cronError) {
    return `Scheduled tasks unavailable: ${params.cronError}`;
  }
  const lines: string[] = [];
  if (params.cronStatus) {
    lines.push(`Cron scheduler: ${params.cronStatus.enabled === false ? "disabled" : "enabled"}`);
    if (params.cronStatus.storePath) {
      lines.push(`Cron store: ${params.cronStatus.storePath}`);
    }
    if (params.cronStatus.nextWakeAtMs) {
      lines.push(`Next scheduler wake: ${new Date(params.cronStatus.nextWakeAtMs).toISOString()}`);
    }
  }
  lines.push(`Scheduled jobs: ${params.cronJobs.length}`);
  if (params.cronJobs.length > 0) {
    lines.push(
      ...params.cronJobs.map((job) => {
        const nextRun = formatIsoTime(job.state?.nextRunAtMs);
        const lastRun = formatIsoTime(job.state?.lastRunAtMs);
        const bits = [
          job.name || job.id || "unnamed job",
          job.id ? `id=${job.id}` : undefined,
          job.enabled === false ? "disabled" : "enabled",
          formatCronSchedule(job),
          nextRun ? `next=${nextRun}` : undefined,
          lastRun ? `last=${lastRun}` : undefined,
          job.state?.lastStatus ? `lastStatus=${job.state.lastStatus}` : undefined,
        ].filter(Boolean);
        return `- ${bits.join(" | ")}`;
      }),
    );
  }
  return lines.join("\n");
}

function formatToolsSection(params: {
  availableToolNames: string[];
  directUserInstruction?: boolean;
}): string {
  const lines = [
    `Available tools (${params.availableToolNames.length}): ${params.availableToolNames.join(", ")}`,
  ];
  if (params.directUserInstruction === false) {
    lines.push(
      "Some self-modifying actions are currently blocked because the instruction is not direct user input.",
    );
  }
  return lines.join("\n");
}

function extractTextBlocks(content: unknown): string {
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .filter(
      (block): block is { type: "text"; text: string } =>
        typeof block === "object" &&
        block !== null &&
        "type" in block &&
        "text" in block &&
        block.type === "text" &&
        typeof block.text === "string",
    )
    .map((block) => block.text.trim())
    .filter(Boolean)
    .join("\n\n");
}

export function createSelfInspectingTool(opts?: {
  agentSessionKey?: string;
  config?: MarvConfig;
  availableToolNames?: string[];
  directUserInstruction?: boolean;
}): AnyAgentTool {
  return {
    label: "Self Inspecting",
    name: "self_inspecting",
    description:
      "Inspect your own runtime, settings, models, scheduled tasks, context, and tool state. Use when the user asks about your current status, settings, available models, scheduled tasks, tools, limits, or why you are behaving a certain way. Optional: cleanupContextPollution=true removes recognized polluted assistant history for the current session.",
    parameters: SelfInspectingToolSchema,
    execute: async (_toolCallId, args) => {
      const cfg = opts?.config ?? loadConfig();
      const sessionKey = opts?.agentSessionKey?.trim();
      if (!sessionKey) {
        throw new ToolInputError("sessionKey required");
      }

      const query = normalizeQuery((args as Record<string, unknown>).query);
      const cleanupRequested = Boolean((args as Record<string, unknown>).cleanupContextPollution);
      if (cleanupRequested && opts?.directUserInstruction === false) {
        return {
          content: [
            {
              type: "text",
              text: "I can't clean my own context from an indirect or forwarded instruction.",
            },
          ],
          details: { ok: false, denied: true, query, cleanupRequested },
        };
      }

      const agentId = resolveAgentIdFromSessionKey(sessionKey);
      const storePath = resolveStorePath(cfg.session?.store, { agentId });
      const store = loadSessionStore(storePath, { skipCache: true });
      const sessionEntry = store[sessionKey];
      const defaultModel = resolveDefaultModelForAgent({ cfg, agentId });
      const currentModel = resolveSessionModelRef(cfg, sessionEntry, agentId);
      const runtimePlan = resolveRuntimeModelPlan({
        cfg,
        agentId,
        agentDir: resolveAgentDir(cfg, agentId),
      });
      const runtimeRegistry = readRuntimeModelRegistry();
      const sessionStatusTool = createSessionStatusTool({
        agentSessionKey: sessionKey,
        config: cfg,
      });
      const sessionStatusResult = await sessionStatusTool.execute("self-inspecting-runtime", {
        sessionKey,
      });
      const sessionStatusText =
        (
          sessionStatusResult.details as {
            statusText?: string;
          }
        )?.statusText ??
        extractTextBlocks(sessionStatusResult.content) ??
        "Session status unavailable.";
      let cronStatus: CronStatusResult | null = null;
      let cronJobs: CronJobSummary[] = [];
      let cronError: string | undefined;
      try {
        cronStatus = await callGatewayTool<CronStatusResult>("cron.status", {}, {});
        const cronList = await callGatewayTool<{ jobs?: CronJobSummary[] }>(
          "cron.list",
          {},
          {
            includeDisabled: true,
          },
        );
        cronJobs = Array.isArray(cronList?.jobs) ? cronList.jobs : [];
      } catch (error) {
        cronError = error instanceof Error ? error.message : String(error);
      }

      const inspection = inspectContextPollution({
        cfg,
        sessionKey,
      });
      const cleanupResult = cleanupRequested
        ? await cleanupContextPollution({
            cfg,
            sessionKey,
          })
        : null;

      const modelsSection = formatModelsSection({
        currentModel,
        defaultModel,
        runtimePlan,
        registry: runtimeRegistry,
      });
      const settingsSection = formatSettingsSection({
        sessionKey,
        sessionEntry,
        directUserInstruction: opts?.directUserInstruction,
      });
      const tasksSection = formatTasksSection({
        cronStatus,
        cronJobs,
        cronError,
      });
      const contextSection = [
        summarizeContextPollution(cleanupResult ?? inspection),
        cleanupResult
          ? `Cleanup removed transcript ${cleanupResult.cleaned.transcriptRemoved}, task context ${cleanupResult.cleaned.taskContextRemoved}.`
          : undefined,
      ]
        .filter(Boolean)
        .join("\n");
      const toolsSection = formatToolsSection({
        availableToolNames: [...new Set(opts?.availableToolNames ?? [])].toSorted(),
        directUserInstruction: opts?.directUserInstruction,
      });

      let text: string;
      switch (query) {
        case "runtime":
          text = sessionStatusText || "Session status unavailable.";
          break;
        case "settings":
          text = settingsSection;
          break;
        case "models":
          text = modelsSection;
          break;
        case "tasks":
          text = tasksSection;
          break;
        case "context":
          text = contextSection;
          break;
        case "tools":
          text = toolsSection;
          break;
        case "all":
          text = [
            "Runtime",
            sessionStatusText || "Session status unavailable.",
            "",
            "Settings",
            settingsSection,
            "",
            "Models",
            modelsSection,
            "",
            "Tasks",
            tasksSection,
            "",
            "Context",
            contextSection,
            "",
            "Tools",
            toolsSection,
          ].join("\n");
          break;
        case "summary":
        default:
          text = [
            `Current model: ${currentModel.provider}/${currentModel.model}`,
            `Default model: ${defaultModel.provider}/${defaultModel.model}`,
            `Model pool: ${runtimePlan.poolName} (${runtimePlan.candidates.length} runnable)`,
            `Scheduled jobs: ${cronJobs.length}`,
            summarizeContextPollution(cleanupResult ?? inspection),
          ].join("\n");
          break;
      }

      return {
        content: [{ type: "text", text }],
        details: {
          ok: true,
          query,
          cleanupRequested,
          cleaned: cleanupResult?.cleaned ?? null,
          runtime: {
            sessionKey,
            agentId,
            currentModel: `${currentModel.provider}/${currentModel.model}`,
            defaultModel: `${defaultModel.provider}/${defaultModel.model}`,
            sessionStatusText,
          },
          settings: {
            modelOverride: sessionEntry?.modelOverride ?? null,
            providerOverride: sessionEntry?.providerOverride ?? null,
            authProfileOverride: sessionEntry?.authProfileOverride ?? null,
            thinkingLevel: sessionEntry?.thinkingLevel ?? null,
            verboseLevel: sessionEntry?.verboseLevel ?? null,
            reasoningLevel: sessionEntry?.reasoningLevel ?? null,
            responseUsage: sessionEntry?.responseUsage ?? null,
            elevatedLevel: sessionEntry?.elevatedLevel ?? null,
            execHost: sessionEntry?.execHost ?? null,
            execSecurity: sessionEntry?.execSecurity ?? null,
            execAsk: sessionEntry?.execAsk ?? null,
            execNode: sessionEntry?.execNode ?? null,
            queueMode: sessionEntry?.queueMode ?? null,
            queueDebounceMs: sessionEntry?.queueDebounceMs ?? null,
            queueCap: sessionEntry?.queueCap ?? null,
            queueDrop: sessionEntry?.queueDrop ?? null,
            updatedAt: sessionEntry?.updatedAt ?? null,
          },
          models: {
            poolName: runtimePlan.poolName,
            candidates: runtimePlan.candidates.map((entry) => entry.ref),
            registryPath: resolveRuntimeRegistryPathForDisplay(),
            registryModelCount: runtimeRegistry?.models.length ?? 0,
            lastSuccessfulRefreshAt: runtimeRegistry?.lastSuccessfulRefreshAt,
            configured: runtimePlan.configured.map((entry) => ({
              ref: entry.ref,
              available: entry.available,
              availabilityReason: entry.availabilityReason,
            })),
          },
          tasks: {
            enabled: cronStatus?.enabled ?? null,
            storePath: cronStatus?.storePath ?? null,
            nextWakeAtMs: cronStatus?.nextWakeAtMs ?? null,
            error: cronError ?? null,
            jobs: cronJobs.map((job) => ({
              id: job.id ?? null,
              name: job.name ?? null,
              enabled: job.enabled ?? null,
              schedule: job.schedule ?? null,
              state: job.state ?? null,
            })),
          },
          context: {
            preferences: (cleanupResult ?? inspection).preferences,
            transcriptViolations: (cleanupResult ?? inspection).transcript.violations.length,
            transcriptRemovable:
              (cleanupResult ?? inspection).transcript.removableIds.length +
              (cleanupResult ?? inspection).transcript.sanitizedIds.length,
            taskContextViolations: (cleanupResult ?? inspection).taskContext.violations.length,
            taskContextRemovable: (cleanupResult ?? inspection).taskContext.removableIds.length,
          },
          tools: {
            available: [...new Set(opts?.availableToolNames ?? [])].toSorted(),
          },
        },
      };
    },
  };
}

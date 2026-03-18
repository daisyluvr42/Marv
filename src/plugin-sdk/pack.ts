/**
 * Profession Pack SDK
 *
 * High-level API for creating industry-specific agent customization packs.
 * Each pack bundles a persona, tools, workflows, and knowledge into a single
 * plugin that can be independently distributed as an npm package.
 *
 * Usage:
 * ```ts
 * import { registerProfessionPack } from "agentmarv/plugin-sdk";
 *
 * export default function register(api: MarvPluginApi) {
 *   registerProfessionPack(api, {
 *     id: "pack-legal",
 *     name: "法律顾问",
 *     persona: { role: "你是一位资深法律顾问助理", domain: "legal" },
 *     tools: [contractReviewTool],
 *     workflows: [contractReviewWorkflow],
 *     knowledge: { dir: "knowledge", autoInject: true },
 *   });
 * }
 * ```
 */

import type {
  MarvPluginApi,
  MarvPluginToolFactory,
  MarvPluginToolOptions,
} from "../plugins/types.js";

// ============================================================================
// Types
// ============================================================================

/** Persona definition controlling the agent's professional identity. */
export type PackPersona = {
  /** System role description, e.g. "你是一位资深法律顾问助理" */
  role: string;
  /** Domain identifier, e.g. "legal", "medical", "finance" */
  domain: string;
  /** Professional constraints and compliance rules */
  constraints?: string[];
  /** Response style guidance for the domain */
  responseStyle?: string;
};

/** A single step in a pack workflow pipeline. */
export type PackWorkflowStep = {
  /** Unique step id within the workflow */
  id: string;
  /** Step type — maps to existing Marv infrastructure */
  type: "llm-task" | "parallel" | "subagent" | "approve" | "tool";
  /** Step configuration (prompt, schema, roles, etc.) */
  config: Record<string, unknown>;
};

/** A reusable workflow template for domain-specific processes. */
export type PackWorkflow = {
  /** Unique workflow id, e.g. "contract-review" */
  id: string;
  /** Human-readable name, e.g. "合同审查" */
  name: string;
  /** Description for agent auto-matching */
  description: string;
  /** Trigger keywords/intents (pipe-separated regex), e.g. "审查合同|review contract" */
  trigger?: string;
  /** Ordered workflow steps */
  steps: PackWorkflowStep[];
};

/** Knowledge base configuration for local vector store. */
export type PackKnowledge = {
  /** Knowledge files directory (relative to pack's source, copied to ~/.marv/packs/<id>/) */
  dir: string;
  /** Auto-inject relevant snippets into context before each agent run (default: true) */
  autoInject?: boolean;
  /** Max snippets to inject per turn (default: 3) */
  maxSnippets?: number;
  /** Minimum similarity score for injection (default: 0.3) */
  minScore?: number;
};

/** Full profession pack definition. */
export type ProfessionPack = {
  /** Pack identifier, e.g. "pack-legal" */
  id: string;
  /** Display name, e.g. "法律顾问" */
  name: string;
  /** Professional persona definition */
  persona: PackPersona;
  /** Optional tool factories for domain-specific tools */
  tools?: MarvPluginToolFactory[];
  /** Optional workflow templates */
  workflows?: PackWorkflow[];
  /** Optional knowledge base configuration */
  knowledge?: PackKnowledge;
};

// ============================================================================
// Constants
// ============================================================================

/** Hook priority for pack persona injection (higher than default plugin priority of 0). */
const PACK_HOOK_PRIORITY = 200;

// ============================================================================
// Persona prompt builder
// ============================================================================

function buildPersonaPrompt(persona: PackPersona): string {
  const lines: string[] = [`<profession-pack domain="${persona.domain}">`, persona.role];

  if (persona.constraints && persona.constraints.length > 0) {
    lines.push("");
    lines.push("Professional constraints:");
    for (const constraint of persona.constraints) {
      lines.push(`- ${constraint}`);
    }
  }

  if (persona.responseStyle) {
    lines.push("");
    lines.push(`Response style: ${persona.responseStyle}`);
  }

  lines.push("</profession-pack>");
  return lines.join("\n");
}

// ============================================================================
// Workflow prompt builder
// ============================================================================

function buildWorkflowsPrompt(workflows: PackWorkflow[]): string {
  if (workflows.length === 0) {
    return "";
  }

  const lines: string[] = [
    "<available-workflows>",
    "The following profession-specific workflows are available. When the user's request matches a workflow trigger or description, use run_workflow to execute it.",
    "",
  ];

  for (const wf of workflows) {
    lines.push(`- ${wf.id}: ${wf.name} — ${wf.description}`);
    if (wf.trigger) {
      lines.push(`  Triggers: ${wf.trigger}`);
    }
  }

  lines.push("</available-workflows>");
  return lines.join("\n");
}

// ============================================================================
// Main registration function
// ============================================================================

/**
 * Register a profession pack with the Marv plugin system.
 *
 * This high-level function wires up persona injection, tools, workflows,
 * and knowledge into the existing plugin hooks and tool registry.
 */
export function registerProfessionPack(api: MarvPluginApi, pack: ProfessionPack): void {
  api.logger.info(`pack: registering profession pack "${pack.name}" (${pack.id})`);

  // 1. Persona injection via before_prompt_build hook
  const personaPrompt = buildPersonaPrompt(pack.persona);
  const workflowsPrompt = pack.workflows ? buildWorkflowsPrompt(pack.workflows) : "";

  api.on(
    "before_prompt_build",
    (_event) => {
      const parts = [personaPrompt];
      if (workflowsPrompt) {
        parts.push(workflowsPrompt);
      }
      return { prependContext: parts.join("\n\n") };
    },
    { priority: PACK_HOOK_PRIORITY },
  );

  // 2. Register domain-specific tools
  if (pack.tools) {
    for (const toolFactory of pack.tools) {
      api.registerTool(toolFactory);
    }
  }

  // 3. Register workflow execution tool if workflows are defined
  if (pack.workflows && pack.workflows.length > 0) {
    registerWorkflowTool(api, pack.workflows);
  }

  // 4. Knowledge base (registered separately if configured)
  // Knowledge tools and hooks are handled by pack-knowledge.ts

  api.logger.info(
    `pack: "${pack.name}" registered — ` +
      `persona: ${pack.persona.domain}, ` +
      `tools: ${pack.tools?.length ?? 0}, ` +
      `workflows: ${pack.workflows?.length ?? 0}`,
  );
}

// ============================================================================
// Workflow tool registration
// ============================================================================

function registerWorkflowTool(api: MarvPluginApi, workflows: PackWorkflow[]): void {
  const workflowMap = new Map<string, PackWorkflow>();
  for (const wf of workflows) {
    workflowMap.set(wf.id, wf);
  }

  const toolOpts: MarvPluginToolOptions = {
    name: "run_workflow",
    optional: true,
  };

  api.registerTool((_ctx) => {
    // Dynamic import to avoid circular deps — Type is from @sinclair/typebox
    // which is already a dependency in the project
    const { Type } = require("@sinclair/typebox");

    return {
      name: "run_workflow",
      label: "Run Workflow",
      description: `Execute a profession-specific workflow by ID. Available workflows: ${workflows.map((w) => `${w.id} (${w.name})`).join(", ")}`,
      parameters: Type.Object({
        workflow_id: Type.String({
          description: `Workflow to run: ${workflows.map((w) => w.id).join(", ")}`,
        }),
        input: Type.Optional(
          Type.Record(Type.String(), Type.Unknown(), {
            description: "Input data for the workflow (e.g. document content, context)",
          }),
        ),
      }),
      async execute(_toolCallId: string, params: Record<string, unknown>) {
        const { workflow_id, input } = params as {
          workflow_id: string;
          input?: Record<string, unknown>;
        };

        const workflow = workflowMap.get(workflow_id);
        if (!workflow) {
          return {
            content: [
              {
                type: "text",
                text: `Unknown workflow "${workflow_id}". Available: ${workflows.map((w) => w.id).join(", ")}`,
              },
            ],
            details: { error: "unknown_workflow", workflow_id },
          };
        }

        // Build step descriptions for the agent to execute sequentially
        const str = (v: unknown, fallback: string) => (typeof v === "string" ? v : fallback);

        const stepDescriptions = workflow.steps.map((step, i) => {
          const stepNum = i + 1;
          switch (step.type) {
            case "llm-task":
              return `Step ${stepNum} (${step.id}): Use llm_task to ${str(step.config.prompt, "process data")}${step.config.schema ? " with structured output" : ""}`;
            case "parallel":
              return `Step ${stepNum} (${step.id}): Use parallel_spawn with roles [${(step.config.roles as string[])?.join(", ") ?? ""}] — task: ${str(step.config.task, "analyze")}`;
            case "subagent":
              return `Step ${stepNum} (${step.id}): Delegate to subagent — ${str(step.config.task, str(step.config.prompt, "process"))}`;
            case "approve":
              return `Step ${stepNum} (${step.id}): Ask user for approval — "${str(step.config.prompt, "Proceed?")}"`;
            case "tool":
              return `Step ${stepNum} (${step.id}): Call tool "${str(step.config.toolName, "unknown")}" with provided input`;
            default:
              return `Step ${stepNum} (${step.id}): ${String(step.type)}`;
          }
        });

        const plan = [
          `Executing workflow: **${workflow.name}** (${workflow.id})`,
          `Description: ${workflow.description}`,
          "",
          "Steps to execute in order:",
          ...stepDescriptions,
          "",
          input ? `Input data: ${JSON.stringify(input)}` : "No input data provided.",
          "",
          "Execute each step sequentially. Pass output from each step as input to the next.",
          "For parallel steps, spawn all roles simultaneously and collect results before proceeding.",
          "For approve steps, present results to the user and wait for confirmation.",
        ].join("\n");

        return {
          content: [{ type: "text", text: plan }],
          details: {
            workflow_id,
            workflow_name: workflow.name,
            step_count: workflow.steps.length,
            steps: workflow.steps.map((s) => ({ id: s.id, type: s.type })),
          },
        };
      },
    };
  }, toolOpts);
}

/**
 * Pack Workflow Engine
 *
 * Converts PackWorkflow definitions into executable pipeline descriptions.
 * Supports both agent-guided execution (via run_workflow tool) and
 * deterministic Eddie pipeline generation for fixed-step workflows.
 */

import type { PackWorkflow, PackWorkflowStep } from "./pack.js";

// ============================================================================
// Eddie Pipeline Generation
// ============================================================================

/**
 * Convert a PackWorkflow into an Eddie pipeline string.
 *
 * Each step type maps to the corresponding marv.invoke action:
 * - llm-task → marv.invoke --action llm_task
 * - parallel → marv.invoke --action parallel_spawn
 * - subagent → marv.invoke --action subagent
 * - approve  → approve --prompt "..."
 * - tool     → marv.invoke --action <toolName>
 *
 * Example output:
 *   marv.invoke --action llm_task --prompt "Extract clauses" |
 *   marv.invoke --action parallel_spawn --roles "analyst,officer" |
 *   marv.invoke --action llm_task --prompt "Synthesize" |
 *   approve --prompt "Send report?"
 */
export function workflowToEddiePipeline(workflow: PackWorkflow): string {
  const stages = workflow.steps.map((step) => stepToEddieStage(step));
  return stages.join(" | \n");
}

/** Safely extract a string config value with fallback. */
function cfg(config: Record<string, unknown>, key: string, fallback: string): string {
  const val = config[key];
  return typeof val === "string" ? val : fallback;
}

function stepToEddieStage(step: PackWorkflowStep): string {
  const c = step.config;
  switch (step.type) {
    case "llm-task": {
      const prompt = escapeArg(cfg(c, "prompt", ""));
      const schemaFlag = c.schema ? " --structured" : "";
      return "marv.invoke --action llm_task --prompt " + prompt + schemaFlag;
    }

    case "parallel": {
      const roles = (c.roles as string[]) ?? [];
      const task = escapeArg(cfg(c, "task", "analyze"));
      return (
        "marv.invoke --action parallel_spawn --roles " +
        escapeArg(roles.join(",")) +
        " --task " +
        task
      );
    }

    case "subagent": {
      const task = escapeArg(cfg(c, "task", cfg(c, "prompt", "process")));
      const role = c.role ? " --role " + escapeArg(cfg(c, "role", "")) : "";
      return "marv.invoke --action subagent --task " + task + role;
    }

    case "approve": {
      const prompt = escapeArg(cfg(c, "prompt", "Proceed?"));
      return "approve --prompt " + prompt;
    }

    case "tool": {
      const toolName = cfg(c, "toolName", "unknown");
      const args = c.args
        ? " " +
          Object.entries(c.args as Record<string, unknown>)
            .map(([k, v]) => "--" + k + " " + escapeArg(String(v)))
            .join(" ")
        : "";
      return "marv.invoke --action " + toolName + args;
    }

    default:
      return "# unknown step type: " + step.type;
  }
}

function escapeArg(value: string): string {
  if (/^[\w.,-]+$/.test(value)) {
    return value;
  }
  return `"${value.replace(/"/g, '\\"')}"`;
}

// ============================================================================
// Workflow Validation
// ============================================================================

export type WorkflowValidationError = {
  stepId: string;
  message: string;
};

/** Validate a workflow definition for common issues. */
export function validateWorkflow(workflow: PackWorkflow): WorkflowValidationError[] {
  const errors: WorkflowValidationError[] = [];
  const stepIds = new Set<string>();

  if (!workflow.id || !/^[\w-]+$/.test(workflow.id)) {
    errors.push({
      stepId: "_root",
      message: `Invalid workflow id "${workflow.id}": must be alphanumeric with hyphens`,
    });
  }

  if (!workflow.steps || workflow.steps.length === 0) {
    errors.push({ stepId: "_root", message: "Workflow must have at least one step" });
    return errors;
  }

  for (const step of workflow.steps) {
    if (stepIds.has(step.id)) {
      errors.push({ stepId: step.id, message: `Duplicate step id "${step.id}"` });
    }
    stepIds.add(step.id);

    const validTypes = ["llm-task", "parallel", "subagent", "approve", "tool"];
    if (!validTypes.includes(step.type)) {
      errors.push({
        stepId: step.id,
        message: `Invalid step type "${step.type}": must be one of ${validTypes.join(", ")}`,
      });
    }

    if (step.type === "parallel" && !step.config.roles) {
      errors.push({ stepId: step.id, message: 'Parallel step requires "roles" in config' });
    }

    if (step.type === "tool" && !step.config.toolName) {
      errors.push({ stepId: step.id, message: 'Tool step requires "toolName" in config' });
    }
  }

  return errors;
}

// ============================================================================
// Workflow Registry Helpers
// ============================================================================

/** Build a summary of available workflows for CLI display. */
export function formatWorkflowList(workflows: PackWorkflow[]): string {
  if (workflows.length === 0) {
    return "No workflows registered.";
  }

  const lines: string[] = [];
  for (const wf of workflows) {
    lines.push(`  ${wf.id}`);
    lines.push(`    ${wf.name} — ${wf.description}`);
    lines.push(`    Steps: ${wf.steps.map((s) => `${s.id}(${s.type})`).join(" → ")}`);
    if (wf.trigger) {
      lines.push(`    Triggers: ${wf.trigger}`);
    }
    lines.push("");
  }
  return lines.join("\n");
}

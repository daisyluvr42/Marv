import type { Command } from "commander";
import {
  taskArchiveCommand,
  taskListCommand,
  taskShowCommand,
  type TaskStatus,
} from "../commands/task.js";
import { setVerbose } from "../globals.js";
import { defaultRuntime } from "../runtime.js";
import { theme } from "../terminal/theme.js";
import { runCommandWithRuntime } from "./cli-utils.js";
import { formatHelpExamples } from "./help-format.js";

function parsePositiveInt(value: string, field: string): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${field} must be a positive integer`);
  }
  return parsed;
}

function parseTaskStatus(value: string): TaskStatus {
  const normalized = value.trim().toLowerCase();
  if (
    normalized === "active" ||
    normalized === "paused" ||
    normalized === "completed" ||
    normalized === "archived"
  ) {
    return normalized;
  }
  throw new Error("status must be one of: active, paused, completed, archived");
}

export function registerTaskCli(program: Command) {
  const task = program.command("task").description("Manage task-context windows and archives");

  task
    .command("list")
    .description("List task contexts for an agent")
    .option("--agent <id>", "Agent id (defaults to configured default)")
    .option("--status <status>", "Filter by task status", parseTaskStatus)
    .option("--limit <n>", "Max tasks to show", (value) => parsePositiveInt(value, "limit"))
    .option("--json", "Output JSON", false)
    .option("--verbose", "Verbose logging", false)
    .addHelpText(
      "after",
      () =>
        `\n${theme.heading("Examples:")}\n${formatHelpExamples([
          ["marv task list", "List recent tasks for the default agent."],
          ["marv task list --status active", "Only show active tasks."],
          ["marv task list --agent work --json", "Machine-readable output for a specific agent."],
        ])}`,
    )
    .action(async (opts) => {
      setVerbose(Boolean(opts.verbose));
      await runCommandWithRuntime(defaultRuntime, async () => {
        await taskListCommand(
          {
            agent: opts.agent as string | undefined,
            status: opts.status as TaskStatus | undefined,
            limit: opts.limit as number | undefined,
            json: Boolean(opts.json),
          },
          defaultRuntime,
        );
      });
    });

  task
    .command("show <taskId>")
    .description("Show task details, decisions, and recent entries")
    .option("--agent <id>", "Agent id (defaults to configured default)")
    .option("--entries <n>", "Entries to include", (value) => parsePositiveInt(value, "entries"))
    .option("--json", "Output JSON", false)
    .option("--verbose", "Verbose logging", false)
    .addHelpText(
      "after",
      () =>
        `\n${theme.heading("Examples:")}\n${formatHelpExamples([
          ["marv task show my-task", "Inspect task metadata and entries."],
          ["marv task show my-task --entries 300", "Show a deeper recent-entry window."],
          ["marv task show my-task --json", "Machine-readable details."],
        ])}`,
    )
    .action(async (taskId: string, opts) => {
      setVerbose(Boolean(opts.verbose));
      await runCommandWithRuntime(defaultRuntime, async () => {
        await taskShowCommand(
          {
            agent: opts.agent as string | undefined,
            taskId,
            entries: opts.entries as number | undefined,
            json: Boolean(opts.json),
          },
          defaultRuntime,
        );
      });
    });

  task
    .command("archive <taskId>")
    .description("Archive a task; by default also distill and inject reusable memory")
    .option("--agent <id>", "Agent id (defaults to configured default)")
    .option("--archive-only", "Only archive, skip distillation/injection", false)
    .option("--json", "Output JSON", false)
    .option("--verbose", "Verbose logging", false)
    .addHelpText(
      "after",
      () =>
        `\n${theme.heading("Examples:")}\n${formatHelpExamples([
          ["marv task archive my-task", "Archive and distill the completed task."],
          ["marv task archive my-task --archive-only", "Archive without memory injection."],
          ["marv task archive my-task --json", "Machine-readable archive summary."],
        ])}`,
    )
    .action(async (taskId: string, opts) => {
      setVerbose(Boolean(opts.verbose));
      await runCommandWithRuntime(defaultRuntime, async () => {
        await taskArchiveCommand(
          {
            agent: opts.agent as string | undefined,
            taskId,
            archiveOnly: Boolean(opts.archiveOnly),
            json: Boolean(opts.json),
          },
          defaultRuntime,
        );
      });
    });
}

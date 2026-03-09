import type { Command } from "commander";
import { defaultRuntime } from "../runtime.js";
import { formatDocsLink } from "../terminal/links.js";
import { theme } from "../terminal/theme.js";
import { inheritOptionFromParent } from "./command-options.js";
import { formatHelpExamples } from "./help-format.js";
import { updateRollbackCommand } from "./update-cli/rollback.js";
import {
  type UpdateCommandOptions,
  type UpdateRollbackOptions,
  type UpdateStatusOptions,
  type UpdateWizardOptions,
} from "./update-cli/shared.js";
import { updateStatusCommand } from "./update-cli/status.js";
import { updateCommand } from "./update-cli/update-command.js";
import { updateWizardCommand } from "./update-cli/wizard.js";

export { updateCommand, updateRollbackCommand, updateStatusCommand, updateWizardCommand };
export type {
  UpdateCommandOptions,
  UpdateRollbackOptions,
  UpdateStatusOptions,
  UpdateWizardOptions,
};

function inheritedUpdateJson(command?: Command): boolean {
  return Boolean(inheritOptionFromParent<boolean>(command, "json"));
}

function inheritedUpdateTimeout(
  opts: { timeout?: unknown },
  command?: Command,
): string | undefined {
  const timeout = opts.timeout as string | undefined;
  if (timeout) {
    return timeout;
  }
  return inheritOptionFromParent<string>(command, "timeout");
}

export function registerUpdateCli(program: Command) {
  const update = program
    .command("update")
    .description("Update Marv and inspect update channel status")
    .option("--json", "Output result as JSON", false)
    .option("--no-restart", "Skip restarting the gateway service after a successful update")
    .option("--channel <stable|beta|dev>", "Persist update channel (git + npm)")
    .option("--tag <dist-tag|version>", "Override npm dist-tag or version for this update")
    .option("--timeout <seconds>", "Timeout for each update step in seconds (default: 1200)")
    .option("--yes", "Skip confirmation prompts (non-interactive)", false)
    .addHelpText("after", () => {
      const examples = [
        ["marv update", "Update a source checkout (git)"],
        ["marv update --channel beta", "Switch to beta channel (git + npm)"],
        ["marv update --channel dev", "Switch to dev channel (git + npm)"],
        ["marv update --tag beta", "One-off update to a dist-tag or version"],
        ["marv update --no-restart", "Update without restarting the service"],
        ["marv update --json", "Output result as JSON"],
        ["marv update --yes", "Non-interactive (accept downgrade prompts)"],
        ["marv update rollback", "Roll back to the last known good git deployment"],
        ["marv update wizard", "Interactive update wizard"],
        ["marv --update", "Shorthand for marv update"],
      ] as const;
      const fmtExamples = examples
        .map(([cmd, desc]) => `  ${theme.command(cmd)} ${theme.muted(`# ${desc}`)}`)
        .join("\n");
      return `
${theme.heading("What this does:")}
  - Git checkouts: fetches, rebases, installs deps, builds, and runs doctor
  - npm installs: updates via detected package manager
  - Rollback: restores the last known good git deployment and optionally restarts the gateway

${theme.heading("Switch channels:")}
  - Use --channel stable|beta|dev to persist the update channel in config
  - Run marv update status to see the active channel and source
  - Use --tag <dist-tag|version> for a one-off npm update without persisting

${theme.heading("Non-interactive:")}
  - Use --yes to accept downgrade prompts
  - Combine with --channel/--tag/--restart/--json/--timeout as needed

${theme.heading("Examples:")}
${fmtExamples}

${theme.heading("Notes:")}
  - Switch channels with --channel stable|beta|dev
  - For global installs: auto-updates via detected package manager when possible (see docs/install/updating.md)
  - Downgrades require confirmation (can break configuration)
  - Skips update if the working directory has uncommitted changes

${theme.muted("Docs:")} ${formatDocsLink("/cli/update", "docs: /cli/update")}`;
    })
    .action(async (opts) => {
      try {
        await updateCommand({
          json: Boolean(opts.json),
          restart: Boolean(opts.restart),
          channel: opts.channel as string | undefined,
          tag: opts.tag as string | undefined,
          timeout: opts.timeout as string | undefined,
          yes: Boolean(opts.yes),
        });
      } catch (err) {
        defaultRuntime.error(String(err));
        defaultRuntime.exit(1);
      }
    });

  update
    .command("rollback")
    .description("Roll back to the last known good git deployment")
    .option("--json", "Output result as JSON", false)
    .option("--no-restart", "Skip restarting the gateway service after a successful rollback")
    .option("--timeout <seconds>", "Timeout for each rollback step in seconds (default: 1200)")
    .addHelpText(
      "after",
      () =>
        `\n${theme.heading("Examples:")}\n${formatHelpExamples([
          ["marv update rollback", "Restore the last known good deployment."],
          ["marv update rollback --no-restart", "Rollback without restarting the service."],
          ["marv update rollback --json", "JSON output."],
        ])}\n\n${theme.heading("Notes:")}\n${theme.muted(
          "- Requires a git deployment with recorded last-known-good state",
        )}\n${theme.muted("- Designed as a local rescue path when the main gateway is unhealthy")}\n\n${theme.muted(
          "Docs:",
        )} ${formatDocsLink("/cli/update", "docs: /cli/update")}`,
    )
    .action(async (opts, command) => {
      try {
        await updateRollbackCommand({
          json: Boolean(opts.json) || inheritedUpdateJson(command),
          restart: Boolean(opts.restart),
          timeout: inheritedUpdateTimeout(opts, command),
        });
      } catch (err) {
        defaultRuntime.error(String(err));
        defaultRuntime.exit(1);
      }
    });

  update
    .command("wizard")
    .description("Interactive update wizard")
    .option("--timeout <seconds>", "Timeout for each update step in seconds (default: 1200)")
    .addHelpText(
      "after",
      `\n${theme.muted("Docs:")} ${formatDocsLink("/cli/update", "docs: /cli/update")}\n`,
    )
    .action(async (opts, command) => {
      try {
        await updateWizardCommand({
          timeout: inheritedUpdateTimeout(opts, command),
        });
      } catch (err) {
        defaultRuntime.error(String(err));
        defaultRuntime.exit(1);
      }
    });

  update
    .command("status")
    .description("Show update channel and version status")
    .option("--json", "Output result as JSON", false)
    .option("--timeout <seconds>", "Timeout for update checks in seconds (default: 3)")
    .addHelpText(
      "after",
      () =>
        `\n${theme.heading("Examples:")}\n${formatHelpExamples([
          ["marv update status", "Show channel + version status."],
          ["marv update status --json", "JSON output."],
          ["marv update status --timeout 10", "Custom timeout."],
        ])}\n\n${theme.heading("Notes:")}\n${theme.muted(
          "- Shows current update channel (stable/beta/dev) and source",
        )}\n${theme.muted("- Includes git tag/branch/SHA for source checkouts")}\n\n${theme.muted(
          "Docs:",
        )} ${formatDocsLink("/cli/update", "docs: /cli/update")}`,
    )
    .action(async (opts, command) => {
      try {
        await updateStatusCommand({
          json: Boolean(opts.json) || inheritedUpdateJson(command),
          timeout: inheritedUpdateTimeout(opts, command),
        });
      } catch (err) {
        defaultRuntime.error(String(err));
        defaultRuntime.exit(1);
      }
    });
}

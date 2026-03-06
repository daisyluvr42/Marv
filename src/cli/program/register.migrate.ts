import type { Command } from "commander";
import { migrateExportCommand, parseMigrateExportOptions } from "../../commands/migrate-export.js";
import { migrateImportCommand, parseMigrateImportOptions } from "../../commands/migrate-import.js";
import { defaultRuntime } from "../../runtime.js";
import { runCommandWithRuntime } from "../cli-utils.js";

export function registerMigrateCommands(program: Command) {
  const migrate = program
    .command("migrate")
    .description("Export or import local data for device migration");

  migrate
    .command("export")
    .description("Create a migration archive from local Marv data")
    .option(
      "--scopes <scopes>",
      "Comma-separated scopes: memory,config,sessions,credentials,workspace,tasks,ledger",
    )
    .option("--format <format>", "plain|encrypted")
    .option("--password <password>", "Archive password for encrypted exports")
    .option("--output <path>", "Write the archive to this path")
    .option("--yes", "Accept default prompts where possible", false)
    .option("--non-interactive", "Disable prompts (requires --scopes)", false)
    .action(async (opts) => {
      await runCommandWithRuntime(defaultRuntime, async () => {
        await migrateExportCommand(defaultRuntime, parseMigrateExportOptions(opts));
      });
    });

  migrate
    .command("import <archive>")
    .description("Import data from a migration archive")
    .option(
      "--scopes <scopes>",
      "Comma-separated scopes to import (default: all scopes in the archive)",
    )
    .option("--password <password>", "Archive password for encrypted imports")
    .option("--force", "Overwrite existing files without prompting", false)
    .option("--dry-run", "Show what would be imported without writing files", false)
    .option("--yes", "Accept default prompts where possible", false)
    .option("--non-interactive", "Disable prompts", false)
    .action(async (archive: string, opts) => {
      await runCommandWithRuntime(defaultRuntime, async () => {
        await migrateImportCommand(defaultRuntime, parseMigrateImportOptions(archive, opts));
      });
    });
}

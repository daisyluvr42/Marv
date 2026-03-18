/**
 * Pack CLI — manage profession packs.
 *
 * Commands:
 *   marv pack install <name>    Install a profession pack from npm
 *   marv pack list               List installed packs
 *   marv pack update <id>        Update a pack and rebuild knowledge index
 *   marv pack knowledge add <id> <file>  Add a file to a pack's knowledge base
 */

import { existsSync, mkdirSync, readdirSync, copyFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, extname, join } from "node:path";
import type { Command } from "commander";

const PACKS_BASE_DIR = join(homedir(), ".marv", "packs");
const SUPPORTED_KNOWLEDGE_EXT = new Set([".md", ".txt", ".text"]);

export function registerPackCli(program: Command): void {
  const pack = program.command("pack").description("Manage profession packs");

  pack
    .command("install")
    .description("Install a profession pack from npm")
    .argument("<name>", "npm package name (e.g. @marv/pack-legal)")
    .action(async (name: string) => {
      const { execSync } = await import("node:child_process");

      console.log(`Installing profession pack: ${name}...`);

      try {
        // Install the pack as a global plugin
        execSync(`npm install --omit=dev ${name}`, {
          cwd: join(homedir(), ".marv"),
          stdio: "inherit",
        });
        console.log(`\nPack "${name}" installed successfully.`);
        console.log("Configure with: marv config set plugins.<pack-id>.config.<key> <value>");
      } catch (err) {
        console.error(`Failed to install pack: ${String(err)}`);
        process.exit(1);
      }
    });

  pack
    .command("list")
    .description("List installed profession packs")
    .action(async () => {
      if (!existsSync(PACKS_BASE_DIR)) {
        console.log("No profession packs installed.");
        return;
      }

      const entries = readdirSync(PACKS_BASE_DIR, { withFileTypes: true });
      const packDirs = entries.filter((e) => e.isDirectory() && e.name.startsWith("pack-"));

      if (packDirs.length === 0) {
        console.log("No profession packs installed.");
        return;
      }

      console.log("Installed profession packs:\n");
      for (const dir of packDirs) {
        const knowledgeDir = join(PACKS_BASE_DIR, dir.name, "knowledge");
        const knowledgeCount = existsSync(knowledgeDir)
          ? readdirSync(knowledgeDir).filter((f) =>
              SUPPORTED_KNOWLEDGE_EXT.has(extname(f).toLowerCase()),
            ).length
          : 0;

        console.log(`  ${dir.name}`);
        console.log(`    Knowledge files: ${knowledgeCount}`);
        console.log(`    Path: ${join(PACKS_BASE_DIR, dir.name)}`);
        console.log("");
      }
    });

  pack
    .command("update")
    .description("Update a pack and rebuild knowledge index")
    .argument("<id>", "Pack ID (e.g. pack-legal)")
    .action(async (id: string) => {
      const packDir = join(PACKS_BASE_DIR, id);
      if (!existsSync(packDir)) {
        console.error(`Pack "${id}" not found at ${packDir}`);
        process.exit(1);
      }

      console.log(`Updating pack "${id}"...`);

      // Delete the knowledge DB to force re-indexing on next load
      const dbDir = join(packDir, "knowledge-db");
      if (existsSync(dbDir)) {
        const { rmSync } = await import("node:fs");
        rmSync(dbDir, { recursive: true, force: true });
        console.log("Knowledge index cleared. Will be rebuilt on next agent start.");
      }

      console.log(`Pack "${id}" updated.`);
    });

  // Knowledge subcommand
  const knowledge = pack.command("knowledge").description("Manage pack knowledge bases");

  knowledge
    .command("add")
    .description("Add a file to a pack's knowledge base")
    .argument("<id>", "Pack ID (e.g. pack-legal)")
    .argument("<file>", "Path to the knowledge file to add")
    .action(async (id: string, file: string) => {
      const knowledgeDir = join(PACKS_BASE_DIR, id, "knowledge");
      mkdirSync(knowledgeDir, { recursive: true });

      const ext = extname(file).toLowerCase();
      if (!SUPPORTED_KNOWLEDGE_EXT.has(ext)) {
        console.error(
          `Unsupported file type "${ext}". Supported: ${[...SUPPORTED_KNOWLEDGE_EXT].join(", ")}`,
        );
        process.exit(1);
      }

      if (!existsSync(file)) {
        console.error(`File not found: ${file}`);
        process.exit(1);
      }

      const targetPath = join(knowledgeDir, basename(file));
      copyFileSync(file, targetPath);
      console.log(`Added "${basename(file)}" to ${id} knowledge base.`);
      console.log("Run 'marv pack update <id>' to rebuild the knowledge index.");
    });

  knowledge
    .command("list")
    .description("List knowledge files in a pack")
    .argument("<id>", "Pack ID (e.g. pack-legal)")
    .action((_id: string) => {
      const knowledgeDir = join(PACKS_BASE_DIR, _id, "knowledge");
      if (!existsSync(knowledgeDir)) {
        console.log(`No knowledge base for pack "${_id}".`);
        return;
      }

      const files = readdirSync(knowledgeDir).filter((f) =>
        SUPPORTED_KNOWLEDGE_EXT.has(extname(f).toLowerCase()),
      );

      if (files.length === 0) {
        console.log(`No knowledge files in pack "${_id}".`);
        return;
      }

      console.log(`Knowledge files for ${_id}:\n`);
      for (const file of files) {
        console.log(`  ${file}`);
      }
    });

  // Workflow subcommand
  const workflow = program.command("workflow").description("Manage pack workflows");

  workflow
    .command("list")
    .description("List all registered workflows from installed packs")
    .action(async () => {
      // Workflows are registered dynamically at runtime via plugin hooks.
      // At CLI time we can only check for pack directories.
      console.log(
        "Workflows are registered at runtime when the agent starts.\n" +
          "Send a message to the agent to see available workflows,\n" +
          "or use the run_workflow tool.",
      );
    });
}

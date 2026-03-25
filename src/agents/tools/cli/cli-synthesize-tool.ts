import fs from "node:fs/promises";
import path from "node:path";
import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../../core/config/config.js";
import { CONFIG_DIR } from "../../../utils.js";
import { assertSandboxPath } from "../../sandbox/sandbox-paths.js";
import { optionalStringEnum, stringEnum } from "../../schema/typebox.js";
import { bumpSkillsSnapshotVersion } from "../../skills/refresh.js";
import { resolveWorkspaceRoot } from "../../workspace-dir.js";
import type { AnyAgentTool } from "../common.js";
import { jsonResult, readStringArrayParam, readStringParam } from "../common.js";
import {
  registerManagedCliManifest,
  resolveManagedCliToolDir,
  updateManagedCliProfileState,
  validateManagedCliProfileId,
} from "./cli-profile-registry.js";
import { verifyManagedCliProfile } from "./cli-profile-runtime.js";
import type { ManagedCliManifest } from "./cli-profile-types.js";

const CliSynthesizeSchema = Type.Object(
  {
    id: Type.String({ minLength: 1 }),
    name: Type.Optional(Type.String()),
    description: Type.String({ minLength: 1 }),
    tier: optionalStringEnum(["script-wrapper", "full-cli"] as const),
    scriptPath: Type.Optional(Type.String()),
    command: Type.Optional(Type.String()),
    interpreter: optionalStringEnum(["bash", "sh", "python3", "python", "node", "bun"] as const),
    staticArgs: Type.Optional(Type.Array(Type.String())),
    argsTemplate: Type.Optional(Type.Array(Type.String())),
    outputMode: Type.Optional(stringEnum(["text", "json", "jsonl"] as const)),
    capabilities: Type.Optional(Type.Array(Type.String())),
    helpArgs: Type.Optional(Type.Array(Type.String())),
    smokeArgs: Type.Optional(Type.Array(Type.String())),
    skipHelpCheck: Type.Optional(Type.Boolean()),
    activate: Type.Optional(Type.Boolean()),
    replace: Type.Optional(Type.Boolean()),
    writesWorkspace: Type.Optional(Type.Boolean()),
    workdir: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

function buildCliSkillDocument(manifest: ManagedCliManifest): string {
  const caps =
    manifest.capabilities && manifest.capabilities.length > 0
      ? `\n## Capabilities\n\n${manifest.capabilities.map((c) => `- ${c}`).join("\n")}\n`
      : "";
  return `---\nname: ${manifest.id}\ndescription: ${manifest.description}\n---\n\n# ${manifest.name}\n\n${manifest.description}\n\n## Usage\n\nInvoke via \`cli_invoke\` with \`profileId="${manifest.id}"\`.${caps}`;
}

function buildReadme(manifest: ManagedCliManifest): string {
  const capabilities = manifest.capabilities?.length
    ? manifest.capabilities.map((value) => `- ${value}`).join("\n")
    : "- (none declared)";
  return `# ${manifest.name}

${manifest.description}

## Command

\`${manifest.entry.command}\`

## Capabilities

${capabilities}
`;
}

function buildManifest(params: {
  id: string;
  name: string;
  description: string;
  tier: "script-wrapper" | "full-cli";
  toolDir: string;
  command: string;
  staticArgs?: string[];
  argsTemplate?: string[];
  outputMode: "text" | "json" | "jsonl";
  capabilities?: string[];
  helpArgs?: string[];
  smokeArgs?: string[];
  writesWorkspace: boolean;
  source: { kind: "script" | "command"; originalPath?: string; originalCommand?: string };
}): ManagedCliManifest {
  const timestamp = new Date().toISOString();
  return {
    manifestVersion: 1,
    id: params.id,
    name: params.name,
    description: params.description,
    version: "0.1.0",
    tier: params.tier,
    toolDir: params.toolDir,
    createdAt: timestamp,
    updatedAt: timestamp,
    entry: {
      command: params.command,
      ...(params.staticArgs && params.staticArgs.length > 0
        ? { staticArgs: params.staticArgs }
        : {}),
      ...(params.argsTemplate && params.argsTemplate.length > 0
        ? { argsTemplate: params.argsTemplate }
        : {}),
      outputMode: params.outputMode,
    },
    ...(params.capabilities && params.capabilities.length > 0
      ? { capabilities: params.capabilities }
      : {}),
    source: params.source,
    policy: {
      writesWorkspace: params.writesWorkspace,
      network: "optional",
      sandboxSafe: false,
    },
    verification: {
      ...(params.helpArgs ? { helpArgs: params.helpArgs } : {}),
      ...(params.smokeArgs ? { smokeArgs: params.smokeArgs } : {}),
    },
    lifecycle: {
      state: "draft",
    },
  };
}

async function copyScriptIntoToolDir(params: {
  scriptPath: string;
  toolDir: string;
}): Promise<string> {
  const scriptFileName = path.basename(params.scriptPath);
  const binDir = path.join(params.toolDir, "bin");
  const targetPath = path.join(binDir, scriptFileName);
  await fs.mkdir(binDir, { recursive: true });
  await fs.copyFile(params.scriptPath, targetPath);
  await fs.chmod(targetPath, 0o755).catch(() => {});
  return `./bin/${scriptFileName}`;
}

export function createCliSynthesizeTool(options?: {
  config?: MarvConfig;
  workspaceDir?: string;
  sandboxed?: boolean;
}): AnyAgentTool | null {
  if (options?.sandboxed === true) {
    return null;
  }
  const workspaceRoot = resolveWorkspaceRoot(options?.workspaceDir);
  return {
    label: "CLI Synthesize",
    name: "cli_synthesize",
    description:
      "Register a new managed CLI profile from a wrapper script or command so the agent can reuse it as a tool.",
    parameters: CliSynthesizeSchema,
    execute: async (_toolCallId, rawArgs) => {
      const args =
        rawArgs && typeof rawArgs === "object" && !Array.isArray(rawArgs)
          ? (rawArgs as Record<string, unknown>)
          : {};
      const id = validateManagedCliProfileId(readStringParam(args, "id", { required: true }));
      const name = readStringParam(args, "name") ?? id;
      const description = readStringParam(args, "description", { required: true });
      const tier =
        (readStringParam(args, "tier") as "script-wrapper" | "full-cli" | undefined) ??
        "script-wrapper";
      const scriptPathRaw = readStringParam(args, "scriptPath");
      const commandRaw = readStringParam(args, "command");
      if (!scriptPathRaw && !commandRaw) {
        throw new Error("Either scriptPath or command is required.");
      }
      const toolDir = resolveManagedCliToolDir(id);
      if (args.replace === true) {
        await fs.rm(toolDir, { recursive: true, force: true }).catch(() => {});
      } else {
        const stat = await fs.stat(toolDir).catch(() => null);
        if (stat) {
          throw new Error(`CLI profile already exists: ${id}. Pass replace=true to overwrite it.`);
        }
      }
      await fs.mkdir(toolDir, { recursive: true });
      let command = commandRaw ?? "";
      let staticArgs = readStringArrayParam(args, "staticArgs") ?? [];
      let source: { kind: "script" | "command"; originalPath?: string; originalCommand?: string };
      if (scriptPathRaw) {
        const resolvedScript = await assertSandboxPath({
          filePath: scriptPathRaw,
          cwd: workspaceRoot,
          root: workspaceRoot,
        });
        const interpreter = readStringParam(args, "interpreter");
        const copiedRelativePath = await copyScriptIntoToolDir({
          scriptPath: resolvedScript.resolved,
          toolDir,
        });
        if (interpreter) {
          command = interpreter;
          staticArgs = [copiedRelativePath, ...staticArgs];
        } else {
          command = copiedRelativePath;
        }
        source = { kind: "script", originalPath: resolvedScript.resolved };
      } else {
        source = { kind: "command", originalCommand: command };
      }
      const helpArgs =
        args.skipHelpCheck === true ? [] : (readStringArrayParam(args, "helpArgs") ?? ["--help"]);
      const smokeArgs = readStringArrayParam(args, "smokeArgs");
      const manifest = buildManifest({
        id,
        name,
        description,
        tier,
        toolDir,
        command,
        staticArgs,
        argsTemplate: readStringArrayParam(args, "argsTemplate"),
        outputMode:
          (readStringParam(args, "outputMode") as "text" | "json" | "jsonl" | undefined) ?? "text",
        capabilities: readStringArrayParam(args, "capabilities"),
        helpArgs,
        smokeArgs,
        writesWorkspace: args.writesWorkspace === true,
        source,
      });
      await fs.writeFile(path.join(toolDir, "README.md"), buildReadme(manifest), "utf8");
      await fs.writeFile(
        path.join(toolDir, "manifest.json"),
        `${JSON.stringify(manifest, null, 2)}\n`,
        "utf8",
      );
      const entry = await registerManagedCliManifest({ manifest });
      const verification = await verifyManagedCliProfile({
        manifest,
        workspaceRoot,
      });
      const updated = await updateManagedCliProfileState({
        profileId: id,
        state: verification.ok ? (args.activate === false ? "verified" : "active") : "draft",
        verificationError: verification.ok ? "" : verification.message,
      });
      // Register a SKILL.md index entry so future sessions can discover this tool.
      if (verification.ok) {
        try {
          const skillDir = path.join(CONFIG_DIR, "skills", id);
          await fs.mkdir(skillDir, { recursive: true });
          await fs.writeFile(
            path.join(skillDir, "SKILL.md"),
            buildCliSkillDocument(manifest),
            "utf8",
          );
          bumpSkillsSnapshotVersion({ reason: "manual" });
        } catch {
          // Non-fatal: CLI profile is usable even if the skill index entry fails.
        }
      }

      return jsonResult({
        ok: verification.ok,
        profileId: id,
        state: updated.entry.state,
        toolDir,
        manifestPath: path.join(toolDir, "manifest.json"),
        verification,
        entry,
      });
    },
  };
}

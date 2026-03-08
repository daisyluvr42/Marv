import fs from "node:fs/promises";
import path from "node:path";
import { CONFIG_DIR } from "../../utils.js";
import { markInstalledSkillUsageRecord } from "../skill-usage-records.js";

const MAX_NAME_LENGTH = 64;
const SKILL_NAME_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export type PersistSynthesizedToolParams = {
  name: string;
  description: string;
  scriptPath: string;
  managedSkillsDir?: string;
};

export type PersistSynthesizedToolResult = {
  ok: boolean;
  skillPath?: string;
  message: string;
};

function resolveManagedSkillsDir(override?: string): string {
  return override ?? path.join(CONFIG_DIR, "skills");
}

function validateSkillName(name: string): string {
  const normalized = name.trim();
  if (!normalized) {
    throw new Error("Skill name is required.");
  }
  if (normalized.length > MAX_NAME_LENGTH) {
    throw new Error(`Skill name must be ${MAX_NAME_LENGTH} characters or fewer.`);
  }
  if (!SKILL_NAME_RE.test(normalized)) {
    throw new Error("Skill name must use hyphen-case letters and numbers only.");
  }
  return normalized;
}

function buildSkillDocument(params: {
  name: string;
  description: string;
  scriptRelativePath: string;
}): string {
  return `---
name: ${params.name}
description: ${params.description}
---

# ${params.name}

Use the synthesized helper at \`${params.scriptRelativePath}\` to handle this workflow.

## Run

\`\`\`bash
${params.scriptRelativePath}
\`\`\`
`;
}

export async function persistSynthesizedTool(
  params: PersistSynthesizedToolParams,
): Promise<PersistSynthesizedToolResult> {
  const name = validateSkillName(params.name);
  const description = params.description.trim();
  if (!description) {
    throw new Error("Skill description is required.");
  }

  const sourceScriptPath = path.resolve(params.scriptPath);
  const scriptSource = await fs.readFile(sourceScriptPath, "utf8");
  const managedSkillsDir = resolveManagedSkillsDir(params.managedSkillsDir);
  const skillDir = path.join(managedSkillsDir, name);
  const scriptsDir = path.join(skillDir, "scripts");
  const scriptFileName = path.basename(sourceScriptPath) || "run.sh";
  const targetScriptPath = path.join(scriptsDir, scriptFileName);
  const scriptRelativePath = path.posix.join("scripts", scriptFileName);

  await fs.mkdir(scriptsDir, { recursive: true });
  await fs.writeFile(targetScriptPath, scriptSource, "utf8");
  await fs.writeFile(
    path.join(skillDir, "SKILL.md"),
    buildSkillDocument({ name, description, scriptRelativePath }),
    "utf8",
  );
  await markInstalledSkillUsageRecord({ skillId: name });

  return {
    ok: true,
    skillPath: skillDir,
    message: `Persisted synthesized tool as managed skill "${name}".`,
  };
}

function parseCliArgs(argv: string[]): Record<string, string> {
  const parsed: Record<string, string> = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token?.startsWith("--")) {
      continue;
    }
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!key || value === undefined || value.startsWith("--")) {
      continue;
    }
    parsed[key] = value;
    index += 1;
  }
  return parsed;
}

async function runCli(argv: string[]): Promise<number> {
  const [command, ...rest] = argv;
  if (command !== "persist") {
    process.stdout.write(
      `${JSON.stringify({ ok: false, message: "Usage: persist --name <name> --description <desc> --script <path>" })}\n`,
    );
    return 1;
  }
  const args = parseCliArgs(rest);
  try {
    const result = await persistSynthesizedTool({
      name: args.name ?? "",
      description: args.description ?? "",
      scriptPath: args.script ?? "",
    });
    process.stdout.write(`${JSON.stringify(result)}\n`);
    return result.ok ? 0 : 1;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stdout.write(`${JSON.stringify({ ok: false, message })}\n`);
    return 1;
  }
}

if (import.meta.main) {
  const exitCode = await runCli(process.argv.slice(2));
  process.exitCode = exitCode;
}

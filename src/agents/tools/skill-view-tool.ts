import fs from "node:fs/promises";
import path from "node:path";
import type { MarvConfig } from "../../core/config/config.js";
import { isPathInside } from "../../security/scan-paths.js";
import { markSkillUsed } from "../skill-usage-records.js";
import { findWorkspaceSkillEntryByName } from "../skills.js";
import { stripSkillFrontmatter } from "../skills/materialize.js";
import type { SkillEntry } from "../skills/types.js";
import type { AnyAgentTool } from "./common.js";
import { jsonResult, readStringParam, ToolInputError } from "./common.js";

const SkillViewSchema = {
  type: "object",
  properties: {
    name: {
      type: "string",
      description: "Skill name from <available_skills>.",
    },
    file: {
      type: "string",
      description: "Optional relative path to a bundled skill resource.",
    },
  },
  required: ["name"],
  additionalProperties: false,
} as const;

async function readSkillResource(entry: SkillEntry, relativeFile?: string) {
  if (!relativeFile) {
    const raw = await fs.readFile(entry.skill.filePath, "utf-8");
    return {
      filePath: entry.skill.filePath,
      content: stripSkillFrontmatter(raw),
    };
  }
  const normalized = relativeFile.replace(/\\/g, "/").trim();
  if (!normalized) {
    throw new ToolInputError("file required");
  }
  const resolved = path.resolve(entry.skill.baseDir, normalized);
  if (!isPathInside(entry.skill.baseDir, resolved)) {
    throw new ToolInputError("file must stay within the skill directory");
  }
  const content = await fs.readFile(resolved, "utf-8");
  return {
    filePath: resolved,
    content,
  };
}

export function createSkillViewTool(options: {
  workspaceDir: string;
  config?: MarvConfig;
  skillFilter?: string[];
}): AnyAgentTool {
  return {
    label: "Skill View",
    name: "skill_view",
    description:
      "Load a skill's canonical instructions or a bundled resource file. Use after selecting a skill from <available_skills>.",
    parameters: SkillViewSchema,
    execute: async (_toolCallId, args) => {
      const params = args as Record<string, unknown>;
      const name = readStringParam(params, "name", { required: true });
      const file = readStringParam(params, "file");
      const entry = findWorkspaceSkillEntryByName(options.workspaceDir, name, {
        config: options.config,
        skillFilter: options.skillFilter,
      });
      if (!entry) {
        throw new ToolInputError(`Unknown skill: ${name}`);
      }
      const resource = await readSkillResource(entry, file);
      await markSkillUsed({ skillId: entry.skill.name }).catch(() => undefined);
      return jsonResult({
        ok: true,
        name: entry.skill.name,
        description: entry.skill.description,
        source: entry.skill.source,
        filePath: resource.filePath,
        baseDir: entry.skill.baseDir,
        content: resource.content,
      });
    },
  };
}

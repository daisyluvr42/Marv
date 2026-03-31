import path from "node:path";
import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../core/config/config.js";
import { CONFIG_DIR } from "../../utils.js";
import { markInstalledSkillUsageRecord, markSkillOutcome } from "../skill-usage-records.js";
import { findWorkspaceSkillEntryByName } from "../skills.js";
import { materializeSkillPackage, type MaterializedSkillFile } from "../skills/materialize.js";
import { resolveSkillTrustLevelForSource } from "../skills/package-metadata.js";
import type { SkillPackageMetadata } from "../skills/types.js";
import type { AnyAgentTool } from "./common.js";
import { jsonResult, readStringParam, ToolInputError } from "./common.js";

const SkillCrystallizeSchema = Type.Object(
  {
    name: Type.String({ minLength: 1 }),
    description: Type.String({ minLength: 1 }),
    body: Type.String({ minLength: 1 }),
    sourceSkill: Type.Optional(Type.String()),
    files: Type.Optional(
      Type.Array(
        Type.Object(
          {
            path: Type.String({ minLength: 1 }),
            content: Type.String(),
          },
          { additionalProperties: false },
        ),
      ),
    ),
    validatedSuccess: Type.Boolean(),
  },
  { additionalProperties: false },
);

function sanitizeSkillDirName(raw: string): string {
  const normalized = raw
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[-.]+|[-.]+$/g, "");
  return normalized || "skill";
}

function resolveCrystallizationTarget(params: {
  workspaceDir: string;
  name: string;
  sourceSkill?: string;
  config?: MarvConfig;
}): {
  targetDir: string;
  existingMetadata: SkillPackageMetadata;
  existingSource?: string;
} {
  const sourceName = params.sourceSkill?.trim() || params.name.trim();
  const existing = findWorkspaceSkillEntryByName(params.workspaceDir, sourceName, {
    config: params.config,
  });
  if (existing) {
    return {
      targetDir: existing.skill.baseDir,
      existingMetadata: existing.packageMetadata ?? {},
      existingSource: existing.skill.source,
    };
  }
  return {
    targetDir: path.join(CONFIG_DIR, "skills", sanitizeSkillDirName(params.name)),
    existingMetadata: {},
  };
}

function readMaterializedFiles(
  params: Record<string, unknown>,
): MaterializedSkillFile[] | undefined {
  const raw = params.files;
  if (!Array.isArray(raw) || raw.length === 0) {
    return undefined;
  }
  const files: MaterializedSkillFile[] = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new ToolInputError("files entries must be objects");
    }
    const record = entry as Record<string, unknown>;
    const filePath = readStringParam(record, "path", { required: true });
    const content = typeof record.content === "string" ? record.content : "";
    files.push({ path: filePath, content });
  }
  return files;
}

export function createSkillCrystallizeTool(options: {
  workspaceDir: string;
  config?: MarvConfig;
}): AnyAgentTool {
  return {
    label: "Skill Crystallize",
    name: "skill_crystallize",
    description:
      "Persist a verified skill improvement by rewriting the canonical installed skill package. Use only after the adapted workflow succeeds.",
    parameters: SkillCrystallizeSchema,
    execute: async (_toolCallId, args) => {
      const params = args as Record<string, unknown>;
      const name = readStringParam(params, "name", { required: true });
      const description = readStringParam(params, "description", { required: true });
      const body = readStringParam(params, "body", { required: true, allowEmpty: true }) ?? "";
      const sourceSkill = readStringParam(params, "sourceSkill");
      const validatedSuccess = params.validatedSuccess === true;
      if (!validatedSuccess) {
        throw new ToolInputError("validatedSuccess must be true before crystallizing a skill");
      }
      const files = readMaterializedFiles(params);
      const target = resolveCrystallizationTarget({
        workspaceDir: options.workspaceDir,
        name,
        sourceSkill,
        config: options.config,
      });
      const metadata: SkillPackageMetadata = {
        source: target.existingMetadata.source ?? target.existingSource,
        originHash: target.existingMetadata.originHash,
        trust:
          target.existingMetadata.trust ??
          resolveSkillTrustLevelForSource(target.existingSource) ??
          "agent-created",
        adaptedFrom: sourceSkill?.trim() || name,
        adaptedAt: new Date().toISOString(),
      };
      const materialized = await materializeSkillPackage({
        targetDir: target.targetDir,
        name,
        description,
        body,
        files,
        metadata,
      });
      await markInstalledSkillUsageRecord({ skillId: name }).catch(() => undefined);
      await markSkillOutcome({
        skillId: name,
        outcome: "success",
      }).catch(() => undefined);
      return jsonResult({
        ok: true,
        name,
        targetDir: materialized.targetDir,
        contentHash: materialized.contentHash,
        replaced: Boolean(target.existingSource),
      });
    },
  };
}

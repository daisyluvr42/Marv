import { Type } from "@sinclair/typebox";
import type { AnyAgentTool } from "../common.js";
import { readStringParam, ToolInputError } from "../common.js";
import {
  handleConfigGet,
  handleConfigSet,
  handleConfigUnset,
  handleSkillList,
  handleSkillSourceAdd,
  handleSkillSourceList,
} from "../config-manage.js";
import {
  readBooleanParam,
  type SelfSettingsArgs,
  type SelfSettingsToolOpts,
} from "./self-settings-normalize.js";

const ConfigManageToolSchema = Type.Object(
  {
    configGet: Type.Optional(
      Type.String({
        description:
          "Read a config value by dot-path (e.g. 'gateway.port', 'tools.web.search.provider', 'models.catalog'). Returns the current value.",
      }),
    ),
    configSet: Type.Optional(
      Type.String({
        description:
          "Set a config value. Format: 'path=value' where path is dot-notation (e.g. 'tools.web.search.provider=tavily', 'gateway.port=4242'). Value is parsed as JSON if valid, otherwise stored as string.",
      }),
    ),
    configUnset: Type.Optional(
      Type.String({
        description: "Remove a config key by dot-path (e.g. 'tools.web.search.tavily.apiKey').",
      }),
    ),
    skillList: Type.Optional(
      Type.Boolean({
        description:
          "List all installed skills, plugins, and managed CLI profiles across all sources.",
      }),
    ),
    skillSourceAdd: Type.Optional(
      Type.String({
        description:
          "Add a skill registry source URL so request_missing_tools can search it for available skills. Format: 'name=url' (e.g. 'community=https://raw.githubusercontent.com/user/skills-registry/main/registry.json').",
      }),
    ),
    skillSourceList: Type.Optional(
      Type.Boolean({
        description: "List all configured skill registry sources.",
      }),
    ),
  },
  { additionalProperties: false },
);

export function createConfigManageTool(opts?: SelfSettingsToolOpts): AnyAgentTool {
  return {
    label: "Config Manage",
    name: "self_settings_config",
    description:
      "Manage configuration and skills: config get/set/unset by dot-path, list installed skills and plugins, and manage skill registry sources.",
    parameters: ConfigManageToolSchema,
    execute: async (_toolCallId, args) => {
      if (!opts?.agentSessionKey?.trim()) {
        throw new ToolInputError("sessionKey required");
      }

      const params = args as SelfSettingsArgs;

      const configGetPath = readStringParam(params, "configGet");
      const configSetExpr = readStringParam(params, "configSet");
      const configUnsetPath = readStringParam(params, "configUnset");
      const skillListFlag = readBooleanParam(params, "skillList");
      const skillSourceAddExpr = readStringParam(params, "skillSourceAdd");
      const skillSourceListFlag = readBooleanParam(params, "skillSourceList");

      const hasAction =
        configGetPath !== undefined ||
        configSetExpr !== undefined ||
        configUnsetPath !== undefined ||
        skillListFlag !== undefined ||
        skillSourceAddExpr !== undefined ||
        skillSourceListFlag !== undefined;

      if (!hasAction) {
        throw new ToolInputError("at least one config or skill action is required");
      }

      const results: Array<{ action: string; result: unknown }> = [];

      if (configGetPath) {
        results.push({ action: "configGet", result: await handleConfigGet(configGetPath) });
      }
      if (configSetExpr) {
        results.push({ action: "configSet", result: await handleConfigSet(configSetExpr) });
      }
      if (configUnsetPath) {
        results.push({ action: "configUnset", result: await handleConfigUnset(configUnsetPath) });
      }
      if (skillListFlag) {
        results.push({ action: "skillList", result: await handleSkillList() });
      }
      if (skillSourceAddExpr) {
        results.push({
          action: "skillSourceAdd",
          result: await handleSkillSourceAdd(skillSourceAddExpr),
        });
      }
      if (skillSourceListFlag) {
        results.push({ action: "skillSourceList", result: await handleSkillSourceList() });
      }

      const summary = results.map((r) => `${r.action}: ${JSON.stringify(r.result)}`).join("\n");
      return {
        content: [{ type: "text" as const, text: summary }],
        details: { ok: true, applied: true, configActions: results },
      };
    },
  };
}

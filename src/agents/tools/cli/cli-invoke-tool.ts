import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../../core/config/config.js";
import { stringEnum } from "../../schema/typebox.js";
import { resolveWorkspaceRoot } from "../../workspace-dir.js";
import type { AnyAgentTool } from "../common.js";
import { jsonResult, readStringArrayParam, readStringParam } from "../common.js";
import { loadManagedCliProfile } from "./cli-profile-registry.js";
import { executeManagedCliProfile } from "./cli-profile-runtime.js";

const CliInvokeSchema = Type.Object(
  {
    profileId: Type.String({ minLength: 1 }),
    templateVarsJson: Type.Optional(Type.String()),
    extraArgs: Type.Optional(Type.Array(Type.String())),
    input: Type.Optional(Type.String()),
    workdir: Type.Optional(Type.String()),
    timeoutSeconds: Type.Optional(Type.Number({ minimum: 10 })),
    captureGitDiff: Type.Optional(Type.Boolean()),
    isolate: Type.Optional(Type.Boolean()),
    kind: Type.Optional(
      stringEnum(["general", "coding"] as const, {
        description: "Task hint. Use coding when the CLI may modify workspace files.",
      }),
    ),
  },
  { additionalProperties: false },
);

export function createCliInvokeTool(options?: {
  config?: MarvConfig;
  workspaceDir?: string;
  sandboxed?: boolean;
}): AnyAgentTool | null {
  if (options?.sandboxed === true) {
    return null;
  }
  const workspaceRoot = resolveWorkspaceRoot(options?.workspaceDir);
  return {
    label: "CLI Invoke",
    name: "cli_invoke",
    description:
      "Invoke a managed synthesized CLI profile that has already been registered with Marv.",
    parameters: CliInvokeSchema,
    execute: async (_toolCallId, rawArgs) => {
      const args =
        rawArgs && typeof rawArgs === "object" && !Array.isArray(rawArgs)
          ? (rawArgs as Record<string, unknown>)
          : {};
      const profileId = readStringParam(args, "profileId", { required: true });
      const record = await loadManagedCliProfile({ profileId });
      if (!record) {
        return jsonResult({
          status: "not_found",
          profileId,
          error: `CLI profile not found: ${profileId}`,
        });
      }
      if (
        record.entry.state === "draft" ||
        record.entry.state === "quarantined" ||
        record.entry.state === "deprecated"
      ) {
        return jsonResult({
          status: "not_ready",
          profileId,
          state: record.entry.state,
          error:
            record.entry.state === "draft"
              ? "CLI profile is still a draft. Verify it before use."
              : record.entry.state === "quarantined"
                ? "CLI profile is quarantined. Re-verify or inspect it before use."
                : "CLI profile is deprecated and should not be used.",
        });
      }
      const kind = readStringParam(args, "kind") ?? "general";
      const result = await executeManagedCliProfile({
        manifest: record.manifest,
        workspaceRoot,
        workdir: readStringParam(args, "workdir"),
        templateVarsJson: readStringParam(args, "templateVarsJson"),
        extraArgs: readStringArrayParam(args, "extraArgs"),
        input: readStringParam(args, "input", { allowEmpty: true }),
        timeoutSeconds: typeof args.timeoutSeconds === "number" ? args.timeoutSeconds : undefined,
        captureGitDiff:
          typeof args.captureGitDiff === "boolean"
            ? args.captureGitDiff
            : kind === "coding" || record.manifest.policy?.writesWorkspace === true,
        isolate: args.isolate === true,
      });
      return jsonResult({
        profileId,
        state: record.entry.state,
        ...result,
      });
    },
  };
}

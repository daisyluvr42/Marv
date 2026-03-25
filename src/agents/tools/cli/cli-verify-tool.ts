import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../../core/config/config.js";
import { resolveWorkspaceRoot } from "../../workspace-dir.js";
import type { AnyAgentTool } from "../common.js";
import { jsonResult, readStringParam } from "../common.js";
import { loadManagedCliProfile, updateManagedCliProfileState } from "./cli-profile-registry.js";
import { verifyManagedCliProfile } from "./cli-profile-runtime.js";

const CliVerifySchema = Type.Object(
  {
    profileId: Type.String({ minLength: 1 }),
    activate: Type.Optional(Type.Boolean()),
    quarantineOnFail: Type.Optional(Type.Boolean()),
    timeoutSeconds: Type.Optional(Type.Number({ minimum: 10 })),
  },
  { additionalProperties: false },
);

export function createCliVerifyTool(options?: {
  config?: MarvConfig;
  workspaceDir?: string;
  sandboxed?: boolean;
}): AnyAgentTool | null {
  if (options?.sandboxed === true) {
    return null;
  }
  const workspaceRoot = resolveWorkspaceRoot(options?.workspaceDir);
  return {
    label: "CLI Verify",
    name: "cli_verify",
    description: "Verify a managed synthesized CLI profile and optionally activate it.",
    parameters: CliVerifySchema,
    execute: async (_toolCallId, rawArgs) => {
      const args =
        rawArgs && typeof rawArgs === "object" && !Array.isArray(rawArgs)
          ? (rawArgs as Record<string, unknown>)
          : {};
      const profileId = readStringParam(args, "profileId", { required: true });
      const record = await loadManagedCliProfile({ profileId });
      if (!record) {
        return jsonResult({
          ok: false,
          profileId,
          error: `CLI profile not found: ${profileId}`,
        });
      }
      const verification = await verifyManagedCliProfile({
        manifest: record.manifest,
        workspaceRoot,
        timeoutSeconds: typeof args.timeoutSeconds === "number" ? args.timeoutSeconds : undefined,
      });
      if (!verification.ok) {
        const updated = await updateManagedCliProfileState({
          profileId,
          state: args.quarantineOnFail === true ? "quarantined" : "draft",
          verificationError: verification.message,
        });
        return jsonResult({
          ok: false,
          profileId,
          state: updated.entry.state,
          verification,
        });
      }
      const updated = await updateManagedCliProfileState({
        profileId,
        state: args.activate === true || record.entry.state === "active" ? "active" : "verified",
        verificationError: "",
      });
      return jsonResult({
        ok: true,
        profileId,
        state: updated.entry.state,
        verification,
      });
    },
  };
}

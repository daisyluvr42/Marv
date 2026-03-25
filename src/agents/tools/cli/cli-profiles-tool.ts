import { Type } from "@sinclair/typebox";
import type { MarvConfig } from "../../../core/config/config.js";
import { stringEnum } from "../../schema/typebox.js";
import type { AnyAgentTool } from "../common.js";
import { jsonResult, readStringParam } from "../common.js";
import {
  listManagedCliProfiles,
  loadManagedCliProfile,
  updateManagedCliProfileState,
} from "./cli-profile-registry.js";

const CliProfilesSchema = Type.Object(
  {
    action: stringEnum(["list", "inspect", "enable", "disable", "quarantine", "retire"] as const),
    profileId: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export function createCliProfilesTool(options?: {
  config?: MarvConfig;
  sandboxed?: boolean;
}): AnyAgentTool | null {
  if (options?.sandboxed === true) {
    return null;
  }
  return {
    label: "CLI Profiles",
    name: "cli_profiles",
    description: "List, inspect, enable, disable, quarantine, or retire managed CLI profiles.",
    parameters: CliProfilesSchema,
    execute: async (_toolCallId, rawArgs) => {
      const args =
        rawArgs && typeof rawArgs === "object" && !Array.isArray(rawArgs)
          ? (rawArgs as Record<string, unknown>)
          : {};
      const action = readStringParam(args, "action", { required: true });
      if (action === "list") {
        const records = await listManagedCliProfiles();
        return jsonResult({
          ok: true,
          profiles: records.map(({ entry }) => ({
            id: entry.id,
            state: entry.state,
            tier: entry.tier,
            name: entry.name,
            description: entry.description,
            capabilities: entry.capabilities,
            updatedAt: entry.updatedAt,
          })),
        });
      }
      const profileId = readStringParam(args, "profileId", { required: true });
      const record = await loadManagedCliProfile({ profileId });
      if (!record) {
        return jsonResult({
          ok: false,
          profileId,
          error: `CLI profile not found: ${profileId}`,
        });
      }
      if (action === "inspect") {
        return jsonResult({
          ok: true,
          profile: {
            entry: record.entry,
            manifest: record.manifest,
          },
        });
      }
      if (action === "enable") {
        if (record.entry.state === "draft" || record.entry.state === "quarantined") {
          return jsonResult({
            ok: false,
            profileId,
            state: record.entry.state,
            error: "Only verified or active profiles can be enabled. Run cli_verify first.",
          });
        }
        const updated = await updateManagedCliProfileState({ profileId, state: "active" });
        return jsonResult({ ok: true, profileId, state: updated.entry.state });
      }
      if (action === "disable") {
        const nextState = record.entry.state === "active" ? "verified" : record.entry.state;
        const updated = await updateManagedCliProfileState({ profileId, state: nextState });
        return jsonResult({ ok: true, profileId, state: updated.entry.state });
      }
      if (action === "quarantine") {
        const updated = await updateManagedCliProfileState({ profileId, state: "quarantined" });
        return jsonResult({ ok: true, profileId, state: updated.entry.state });
      }
      if (action === "retire") {
        const updated = await updateManagedCliProfileState({ profileId, state: "deprecated" });
        return jsonResult({ ok: true, profileId, state: updated.entry.state });
      }
      return jsonResult({
        ok: false,
        profileId,
        error: `Unsupported action: ${action}`,
      });
    },
  };
}

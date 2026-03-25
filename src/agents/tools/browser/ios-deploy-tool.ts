import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Type } from "@sinclair/typebox";
import { runExec } from "../../../process/exec.js";
import { optionalStringEnum } from "../../schema/typebox.js";
import type { AnyAgentTool } from "../common.js";
import { jsonResult, readStringParam } from "../common.js";

const IosDeployToolSchema = Type.Object({
  action: optionalStringEnum(["deploy", "list-devices"] as const, {
    description:
      "Action to perform. 'deploy' builds, installs, and launches the iOS app on a connected iPhone. 'list-devices' lists connected iOS devices.",
  }),
  device: Type.Optional(
    Type.String({
      description: "Target device name or UDID. Omit to use the first connected iPhone.",
    }),
  ),
  configuration: Type.Optional(
    Type.String({
      description: "Build configuration: Debug (default) or Release.",
    }),
  ),
  skipBuild: Type.Optional(
    Type.Boolean({
      description: "Skip the build step and install the previously built .app bundle.",
    }),
  ),
});

/**
 * Resolve the path to `scripts/ios-deploy.sh` relative to the package root.
 * Works in dev (src/) and in built output (dist/).
 */
function resolveIosDeployScript(): string | undefined {
  try {
    const moduleDir = path.dirname(fileURLToPath(import.meta.url));
    // dev: src/agents/tools/ → ../../.. → repo root
    // dist: dist/agents/tools/ → ../../.. → repo root (npm install)
    for (const rel of ["../../..", "../.."]) {
      const candidate = path.resolve(moduleDir, rel, "scripts", "ios-deploy.sh");
      if (fs.existsSync(candidate)) {
        return candidate;
      }
    }
  } catch {
    // ignore
  }
  return undefined;
}

export function createIosDeployTool(): AnyAgentTool {
  return {
    label: "iOS Deploy",
    name: "ios_deploy",
    description:
      "Build and deploy the MarvCompanion iOS app to a connected iPhone, or list connected iOS devices. " +
      "Requires Xcode and a USB-connected iPhone with developer mode enabled.",
    parameters: IosDeployToolSchema,
    execute: async (_toolCallId, args) => {
      const params = (args ?? {}) as Record<string, unknown>;
      const action = readStringParam(params, "action") || "deploy";
      const device = readStringParam(params, "device");
      const configuration = readStringParam(params, "configuration");
      const skipBuild = params.skipBuild === true;

      const scriptPath = resolveIosDeployScript();
      if (!scriptPath) {
        return jsonResult({
          status: "error",
          error: "ios-deploy.sh script not found. Ensure the Marv repo is intact.",
        });
      }

      const scriptArgs: string[] = [];

      if (action === "list-devices") {
        scriptArgs.push("--list-devices");
      } else {
        if (device) {
          scriptArgs.push("--device", device);
        }
        if (configuration) {
          scriptArgs.push("--configuration", configuration);
        }
        if (skipBuild) {
          scriptArgs.push("--skip-build");
        }
      }

      try {
        // Build can take several minutes; allow up to 5 minutes.
        const { stdout, stderr } = await runExec("bash", [scriptPath, ...scriptArgs], {
          timeoutMs: 300_000,
          maxBuffer: 10 * 1024 * 1024,
        });

        // The script outputs JSON on stdout for machine consumption.
        const trimmed = stdout.trim();
        try {
          const parsed = JSON.parse(trimmed);
          return jsonResult(parsed);
        } catch {
          // Non-JSON output (e.g., device list or build logs).
          return jsonResult({
            status: "ok",
            output: trimmed,
            logs: stderr.trim() || undefined,
          });
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        // Try to extract JSON error from stderr.
        const stderrMatch = message.match(/\{"status":"error".*\}/);
        if (stderrMatch) {
          try {
            return jsonResult(JSON.parse(stderrMatch[0]));
          } catch {
            // fall through
          }
        }
        return jsonResult({
          status: "error",
          error: message.slice(0, 500),
        });
      }
    },
  };
}

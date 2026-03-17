import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { MarvConfig, PluginRuntime } from "agentmarv/plugin-sdk";
import type { DynamicAgentCreationConfig } from "./types.js";

export type MaybeCreateDynamicAgentResult = {
  created: boolean;
  updatedCfg: MarvConfig;
  agentId?: string;
};

/**
 * Check if a dynamic agent should be created for a DM user and create it if needed.
 * This creates a unique agent instance with its own workspace for each DM user.
 */
export async function maybeCreateDynamicAgent(params: {
  cfg: MarvConfig;
  runtime: PluginRuntime;
  senderOpenId: string;
  dynamicCfg: DynamicAgentCreationConfig;
  log: (msg: string) => void;
}): Promise<MaybeCreateDynamicAgentResult> {
  const { cfg, runtime, senderOpenId, dynamicCfg, log } = params;

  // With single-agent model, check if defaults already exist
  const agent = cfg.agents?.defaults;
  if (agent) {
    return { created: false, updatedCfg: cfg };
  }

  // Use full OpenID as agent ID suffix (OpenID format: ou_xxx is already filesystem-safe)
  const agentId = `feishu-${senderOpenId}`;

  // Resolve path templates with substitutions
  const workspaceTemplate = dynamicCfg.workspaceTemplate ?? "~/.marv/workspace-{agentId}";
  const agentDirTemplate = dynamicCfg.agentDirTemplate ?? "~/.marv/agents/{agentId}/agent";

  const workspace = resolveUserPath(
    workspaceTemplate.replace("{userId}", senderOpenId).replace("{agentId}", agentId),
  );
  const agentDir = resolveUserPath(
    agentDirTemplate.replace("{userId}", senderOpenId).replace("{agentId}", agentId),
  );

  log(`feishu: creating dynamic agent "${agentId}" for user ${senderOpenId}`);
  log(`  workspace: ${workspace}`);
  log(`  agentDir: ${agentDir}`);

  // Create directories
  await fs.promises.mkdir(workspace, { recursive: true });
  await fs.promises.mkdir(agentDir, { recursive: true });

  // Update configuration with new agent defaults
  const updatedCfg: MarvConfig = {
    ...cfg,
    agents: {
      ...cfg.agents,
      defaults: { ...cfg.agents?.defaults, workspace, agentDir },
    },
  };

  // Write updated config using PluginRuntime API
  await runtime.config.writeConfigFile(updatedCfg);

  return { created: true, updatedCfg, agentId };
}

/**
 * Resolve a path that may start with ~ to the user's home directory.
 */
function resolveUserPath(p: string): string {
  if (p.startsWith("~/")) {
    return path.join(os.homedir(), p.slice(2));
  }
  return p;
}

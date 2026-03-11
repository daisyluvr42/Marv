import type { AgentDefaultsConfig, ModelPoolConfig } from "./types.agent-defaults.js";

export type AgentsConfig = {
  defaults?: AgentDefaultsConfig;
  modelPools?: Record<string, ModelPoolConfig>;
};

import type { MarvConfig } from "../config/config.js";

export function applyOnboardingLocalWorkspaceConfig(
  baseConfig: MarvConfig,
  workspaceDir: string,
): MarvConfig {
  return {
    ...baseConfig,
    agents: {
      ...baseConfig.agents,
      defaults: {
        ...baseConfig.agents?.defaults,
        workspace: workspaceDir,
      },
    },
    gateway: {
      ...baseConfig.gateway,
      mode: "local",
    },
  };
}

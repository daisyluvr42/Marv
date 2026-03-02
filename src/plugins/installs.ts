import type { MarvConfig } from "../core/config/config.js";
import type { PluginInstallRecord } from "../core/config/types.plugins.js";

export type PluginInstallUpdate = PluginInstallRecord & { pluginId: string };

export function recordPluginInstall(cfg: MarvConfig, update: PluginInstallUpdate): MarvConfig {
  const { pluginId, ...record } = update;
  const installs = {
    ...cfg.plugins?.installs,
    [pluginId]: {
      ...cfg.plugins?.installs?.[pluginId],
      ...record,
      installedAt: record.installedAt ?? new Date().toISOString(),
    },
  };

  return {
    ...cfg,
    plugins: {
      ...cfg.plugins,
      installs: {
        ...installs,
        [pluginId]: installs[pluginId],
      },
    },
  };
}

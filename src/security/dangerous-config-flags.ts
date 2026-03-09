import type { MarvConfig } from "../core/config/config.js";

function readBooleanFlag(source: unknown, key: string): boolean {
  if (!source || typeof source !== "object") {
    return false;
  }
  return (source as Record<string, unknown>)[key] === true;
}

export function collectEnabledInsecureOrDangerousFlags(cfg: MarvConfig): string[] {
  const enabledFlags: string[] = [];

  if (cfg.gateway?.controlUi?.allowInsecureAuth === true) {
    enabledFlags.push("gateway.controlUi.allowInsecureAuth=true");
  }
  if (readBooleanFlag(cfg.gateway?.controlUi, "dangerouslyAllowHostHeaderOriginFallback")) {
    enabledFlags.push("gateway.controlUi.dangerouslyAllowHostHeaderOriginFallback=true");
  }
  if (cfg.gateway?.controlUi?.dangerouslyDisableDeviceAuth === true) {
    enabledFlags.push("gateway.controlUi.dangerouslyDisableDeviceAuth=true");
  }
  if (readBooleanFlag(cfg.gateway, "allowRealIpFallback")) {
    enabledFlags.push("gateway.allowRealIpFallback=true");
  }
  if (cfg.hooks?.gmail?.allowUnsafeExternalContent === true) {
    enabledFlags.push("hooks.gmail.allowUnsafeExternalContent=true");
  }
  if (Array.isArray(cfg.hooks?.mappings)) {
    for (const [index, mapping] of cfg.hooks.mappings.entries()) {
      if (mapping?.allowUnsafeExternalContent === true) {
        enabledFlags.push(`hooks.mappings[${index}].allowUnsafeExternalContent=true`);
      }
    }
  }
  if (cfg.tools?.exec?.applyPatch?.workspaceOnly === false) {
    enabledFlags.push("tools.exec.applyPatch.workspaceOnly=false");
  }

  return enabledFlags;
}

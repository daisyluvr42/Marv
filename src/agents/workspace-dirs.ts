import type { MarvConfig } from "../core/config/config.js";
import { resolveAgentWorkspaceDir, resolveDefaultAgentId } from "./agent-scope.js";

export function listAgentWorkspaceDirs(cfg: MarvConfig): string[] {
  return [resolveAgentWorkspaceDir(cfg, resolveDefaultAgentId(cfg))];
}

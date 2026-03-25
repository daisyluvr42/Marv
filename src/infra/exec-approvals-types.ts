/**
 * Shared exec-approvals types extracted to break the circular dependency
 * between exec-approvals.ts and exec-approvals-allowlist.ts.
 */

export type ExecAllowlistEntry = {
  id?: string;
  pattern: string;
  lastUsedAt?: number;
  lastUsedCommand?: string;
  lastResolvedPath?: string;
};

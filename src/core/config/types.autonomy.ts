import type { ToolProfileId } from "./types.tools.js";

/** Categories of sensitive information that should be protected by the privacy guard. */
export type PrivacyCategory =
  | "api_keys"
  | "passwords"
  | "tokens"
  | "private_keys"
  | "env_secrets"
  | "personal_info"
  | "internal_urls"
  | "config_secrets";

/** All privacy categories for default-all behavior. */
export const ALL_PRIVACY_CATEGORIES: readonly PrivacyCategory[] = [
  "api_keys",
  "passwords",
  "tokens",
  "private_keys",
  "env_secrets",
  "personal_info",
  "internal_urls",
  "config_secrets",
] as const;

/** Autonomy mode levels. */
export type AutonomyMode = "full" | "supervised" | "minimal";

/** Skills loading strategy under autonomy. */
export type AutonomySkillsMode = "all" | "eligible" | "manual";

/** Scope for tool discovery search. */
export type AutonomyDiscoveryScope = "bundled" | "managed" | "all";

/** Approval granularity for auto-install. */
export type AutonomyInstallApproval = "per-skill" | "batch";

/** Privacy guard configuration. */
export type AutonomyPrivacyConfig = {
  /** Enable strict privacy guard in non-owner / group contexts. Default: true. */
  enabled?: boolean;
  /** Categories of information to protect. Default: all categories. */
  categories?: PrivacyCategory[];
  /** Scan agent output before sending and redact detected secrets. Default: true. */
  outputScan?: boolean;
};

/** Permission escalation configuration. */
export type AutonomyEscalationConfig = {
  /** Allow the agent to request elevated permissions. Default: true. */
  enabled?: boolean;
  /** Escalated permissions are automatically revoked when the task ends (always true). */
  taskScoped?: true;
  /** Timeout for user approval prompts in seconds. Default: 120. */
  approvalTimeoutSeconds?: number;
};

/** Tool discovery configuration. */
export type AutonomyDiscoveryConfig = {
  /** Enable proactive tool/skill discovery. Default: true. */
  enabled?: boolean;
  /** Search scope for skill discovery. Default: "all". */
  scope?: AutonomyDiscoveryScope;
  /** Approval granularity when auto-installing discovered skills. Default: "per-skill". */
  installApproval?: AutonomyInstallApproval;
};

/**
 * Top-level autonomy configuration.
 *
 * Controls how autonomous the agent is by default: which skills are loaded,
 * whether it can discover and install missing tools, how privileges escalate,
 * and how privacy is enforced in non-private contexts.
 */
export type AutonomyConfig = {
  /** Autonomy level. Default: "full". */
  mode?: AutonomyMode;
  /**
   * Execution approval strictness.
   * - "relaxed" (default for personal assistant): allows most commands implicitly.
   * - "strict": intercepts unknown mutating commands but heuristically allows safe commands.
   */
  approvalMode?: "relaxed" | "strict";
  /** Skills loading strategy. Default: "all". */
  skills?: AutonomySkillsMode;
  /** Override the tool profile when autonomy is active. */
  toolProfile?: ToolProfileId;
  /** Allow the agent to auto-install missing skills (with user approval). Default: true. */
  autoInstallSkills?: boolean;
  /** Permission escalation (task-scoped). */
  escalation?: AutonomyEscalationConfig;
  /** Proactive tool/skill discovery. */
  discovery?: AutonomyDiscoveryConfig;
  /** Privacy guard for non-owner / group chat contexts. */
  privacy?: AutonomyPrivacyConfig;
};

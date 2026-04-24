/**
 * AgentGateway — boundary interface between auto-reply/execution and agents/.
 *
 * auto-reply consumers import from this module instead of reaching into agent
 * internals.  The agents module provides a concrete implementation via
 * `createAgentGateway()` in `agents/gateway.ts`.
 *
 * Type-only imports from agents/ are still allowed — they carry no runtime
 * coupling.
 */

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/** Thin facade over the embedded Pi runner + CLI runner + model-fallback loop. */
export interface AgentRunner {
  /** Run the embedded Pi agent (delegates to `runEmbeddedPiAgent`). */
  runEmbedded: (
    ...args: Parameters<typeof import("../../agents/runner/pi-embedded.js").runEmbeddedPiAgent>
  ) => ReturnType<typeof import("../../agents/runner/pi-embedded.js").runEmbeddedPiAgent>;

  /** Run a CLI-backed agent (delegates to `runCliAgent`). */
  runCli: (
    ...args: Parameters<typeof import("../../agents/runner/cli-runner.js").runCliAgent>
  ) => ReturnType<typeof import("../../agents/runner/cli-runner.js").runCliAgent>;

  /** Run with automatic model fallback (delegates to `runWithModelFallback`). */
  runWithFallback: typeof import("../../agents/model/model-fallback.js").runWithModelFallback;

  /** Queue a steering message into an in-flight embedded run. */
  queueEmbeddedMessage: typeof import("../../agents/runner/pi-embedded.js").queueEmbeddedPiMessage;

  /** Abort an active embedded run. */
  abortEmbeddedRun: typeof import("../../agents/runner/pi-embedded.js").abortEmbeddedPiRun;

  /** Check whether an embedded run is currently active. */
  isEmbeddedRunActive: typeof import("../../agents/runner/pi-embedded.js").isEmbeddedPiRunActive;

  /** Check whether an embedded run is currently streaming. */
  isEmbeddedRunStreaming: typeof import("../../agents/runner/pi-embedded.js").isEmbeddedPiRunStreaming;

  /** Resolve the command-queue lane key for an embedded session. */
  resolveEmbeddedSessionLane: typeof import("../../agents/runner/pi-embedded.js").resolveEmbeddedSessionLane;
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

export interface AgentErrorHelpers {
  isContextOverflowError: typeof import("../../agents/runner/pi-embedded-helpers.js").isContextOverflowError;
  isLikelyContextOverflowError: typeof import("../../agents/runner/pi-embedded-helpers.js").isLikelyContextOverflowError;
  isCompactionFailureError: typeof import("../../agents/runner/pi-embedded-helpers.js").isCompactionFailureError;
  isTransientHttpError: typeof import("../../agents/runner/pi-embedded-helpers.js").isTransientHttpError;
  sanitizeUserFacingText: typeof import("../../agents/runner/pi-embedded-helpers.js").sanitizeUserFacingText;
}

// ---------------------------------------------------------------------------
// Model resolution
// ---------------------------------------------------------------------------

export interface AgentModelResolution {
  /** Check whether a provider string refers to a CLI-backed provider. */
  isCliProvider: typeof import("../../agents/model/model-resolve.js").isCliProvider;

  /** Lookup the context-window token limit for a model. */
  lookupContextTokens: typeof import("../../agents/context.js").lookupContextTokens;

  /** Load the full model catalog. */
  loadModelCatalog: typeof import("../../agents/model/model-catalog.js").loadModelCatalog;

  /** Check if there are explicit model selections configured. */
  hasConfiguredModelSelections: typeof import("../../agents/model/model-selections-store.js").hasConfiguredModelSelections;

  /** Resolve the auth mode for a provider (api-key / oauth / etc.). */
  resolveModelAuthMode: typeof import("../../agents/model/model-auth.js").resolveModelAuthMode;

  /** Resolve a runtime model plan (pool, candidates, capabilities). */
  resolveRuntimeModelPlan: typeof import("../../agents/model/model-pool.js").resolveRuntimeModelPlan;

  /** Reorder model candidates by thinking-model preferences. */
  applyThinkingModelPreferences: typeof import("../../agents/model/model-pool.js").applyThinkingModelPreferences;

  /** Build the set of allowed model keys from config + catalog. */
  buildAllowedModelSet: typeof import("../../agents/model/model-resolve.js").buildAllowedModelSet;

  /** Canonical key for a provider/model pair. */
  modelKey: typeof import("../../agents/model/model-resolve.js").modelKey;

  /** Normalize a provider ID string. */
  normalizeProviderId: typeof import("../../agents/model/model-resolve.js").normalizeProviderId;

  /** Resolve a raw model string into a structured ref. */
  resolveModelRefFromString: typeof import("../../agents/model/model-resolve.js").resolveModelRefFromString;

  /** Resolve the default thinking level for a model from catalog. */
  resolveThinkingDefault: typeof import("../../agents/model/model-resolve.js").resolveThinkingDefault;
}

// ---------------------------------------------------------------------------
// Auth profiles
// ---------------------------------------------------------------------------

export interface AgentAuthProfiles {
  /** Clear a session-level auth profile override. */
  clearSessionAuthProfileOverride: typeof import("../../agents/auth-profiles/session-override.js").clearSessionAuthProfileOverride;

  /** Resolve the active auth profile override for a session. */
  resolveSessionAuthProfileOverride: typeof import("../../agents/auth-profiles/session-override.js").resolveSessionAuthProfileOverride;

  /** Get the CLI session ID from a session entry. */
  getCliSessionId: typeof import("../../agents/cli-session.js").getCliSessionId;

  /** Lazily load the auth profile store (avoids circular deps at import time). */
  ensureAuthProfileStore: typeof import("../../agents/auth-profiles.js").ensureAuthProfileStore;
}

// ---------------------------------------------------------------------------
// Sandbox
// ---------------------------------------------------------------------------

export interface AgentSandbox {
  /** Resolve the sandbox runtime status for a session. */
  resolveSandboxRuntimeStatus: typeof import("../../agents/sandbox/sandbox.js").resolveSandboxRuntimeStatus;

  /** Resolve sandbox config for a specific agent. */
  resolveSandboxConfigForAgent: typeof import("../../agents/sandbox/sandbox.js").resolveSandboxConfigForAgent;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export interface AgentConstants {
  DEFAULT_CONTEXT_TOKENS: number;
  DEFAULT_PI_COMPACTION_RESERVE_TOKENS_FLOOR: number;
}

// ---------------------------------------------------------------------------
// Misc utilities
// ---------------------------------------------------------------------------

export interface AgentUtils {
  /** Resolve a cron-style "Current time:" line for prompts. */
  resolveCronStyleNow: typeof import("../../agents/current-time.js").resolveCronStyleNow;

  /** Check if usage has any nonzero values. */
  hasNonzeroUsage: typeof import("../../agents/usage.js").hasNonzeroUsage;
}

// ---------------------------------------------------------------------------
// Composite gateway
// ---------------------------------------------------------------------------

export interface AgentGateway {
  runner: AgentRunner;
  errors: AgentErrorHelpers;
  models: AgentModelResolution;
  auth: AgentAuthProfiles;
  sandbox: AgentSandbox;
  constants: AgentConstants;
  utils: AgentUtils;
}

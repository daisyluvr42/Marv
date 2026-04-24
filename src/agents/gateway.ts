/**
 * Concrete AgentGateway — single-point facade for all agent functions
 * consumed by auto-reply/execution/.
 *
 * Instead of importing from 15+ agent submodules, execution/ files import
 * `{ agents }` from here and access `agents.runner.*`, `agents.models.*`,
 * etc.  When agent internals move/rename, only this file needs updating.
 */

import type { AgentGateway } from "../auto-reply/execution/agent-gateway.js";
// ---------------------------------------------------------------------------
// Auth profiles
// ---------------------------------------------------------------------------
import {
  clearSessionAuthProfileOverride,
  resolveSessionAuthProfileOverride,
} from "./auth-profiles/session-override.js";
import { ensureAuthProfileStore } from "./auth-profiles/store.js";
import { getCliSessionId } from "./cli-session.js";
import { lookupContextTokens } from "./context.js";
// ---------------------------------------------------------------------------
// Misc utils
// ---------------------------------------------------------------------------
import { resolveCronStyleNow } from "./current-time.js";
// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
import { DEFAULT_CONTEXT_TOKENS } from "./defaults.js";
import { resolveModelAuthMode } from "./model/model-auth.js";
import { loadModelCatalog } from "./model/model-catalog.js";
import { runWithModelFallback } from "./model/model-fallback.js";
import { resolveRuntimeModelPlan, applyThinkingModelPreferences } from "./model/model-pool.js";
// ---------------------------------------------------------------------------
// Model resolution
// ---------------------------------------------------------------------------
import { isCliProvider } from "./model/model-resolve.js";
import {
  buildAllowedModelSet,
  modelKey,
  normalizeProviderId,
  resolveModelRefFromString,
  resolveThinkingDefault,
} from "./model/model-resolve.js";
import { hasConfiguredModelSelections } from "./model/model-selections-store.js";
import { runCliAgent } from "./runner/cli-runner.js";
// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------
import {
  isCompactionFailureError,
  isContextOverflowError,
  isLikelyContextOverflowError,
  isTransientHttpError,
  sanitizeUserFacingText,
} from "./runner/pi-embedded-helpers.js";
// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------
import {
  abortEmbeddedPiRun,
  isEmbeddedPiRunActive,
  isEmbeddedPiRunStreaming,
  queueEmbeddedPiMessage,
  resolveEmbeddedSessionLane,
  runEmbeddedPiAgent,
} from "./runner/pi-embedded.js";
import { DEFAULT_PI_COMPACTION_RESERVE_TOKENS_FLOOR } from "./runner/pi-settings.js";
// ---------------------------------------------------------------------------
// Sandbox
// ---------------------------------------------------------------------------
import { resolveSandboxConfigForAgent, resolveSandboxRuntimeStatus } from "./sandbox/sandbox.js";
import { hasNonzeroUsage } from "./usage.js";

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

const _gateway: AgentGateway = {
  runner: {
    runEmbedded: runEmbeddedPiAgent,
    runCli: runCliAgent,
    runWithFallback: runWithModelFallback,
    queueEmbeddedMessage: queueEmbeddedPiMessage,
    abortEmbeddedRun: abortEmbeddedPiRun,
    isEmbeddedRunActive: isEmbeddedPiRunActive,
    isEmbeddedRunStreaming: isEmbeddedPiRunStreaming,
    resolveEmbeddedSessionLane,
  },

  errors: {
    isContextOverflowError,
    isLikelyContextOverflowError,
    isCompactionFailureError,
    isTransientHttpError,
    sanitizeUserFacingText,
  },

  models: {
    isCliProvider,
    lookupContextTokens,
    loadModelCatalog,
    hasConfiguredModelSelections,
    resolveModelAuthMode,
    resolveRuntimeModelPlan,
    applyThinkingModelPreferences,
    buildAllowedModelSet,
    modelKey,
    normalizeProviderId,
    resolveModelRefFromString,
    resolveThinkingDefault,
  },

  auth: {
    clearSessionAuthProfileOverride,
    resolveSessionAuthProfileOverride,
    getCliSessionId,
    ensureAuthProfileStore,
  },

  sandbox: {
    resolveSandboxRuntimeStatus,
    resolveSandboxConfigForAgent,
  },

  constants: {
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_PI_COMPACTION_RESERVE_TOKENS_FLOOR,
  },

  utils: {
    resolveCronStyleNow,
    hasNonzeroUsage,
  },
};

/**
 * Return the AgentGateway singleton.
 *
 * auto-reply/execution files call this to access agent capabilities through
 * a single, well-defined import path.
 */
export function getAgentGateway(): AgentGateway {
  return _gateway;
}

/**
 * Convenience alias for module-level destructuring:
 *   `import { agents } from "../../agents/gateway.js"`
 */
export const agents = _gateway;

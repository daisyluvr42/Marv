---
title: "Model Pool Routing Design"
summary: "Replace layered model routing with configured-model pools, user model overrides, and a single ordered candidate planner."
---

# Model Pool Routing Design

## Goal

Replace the current multi-layer model selection path with a single model-pool planner that is easier to reason about, easier to debug, and predictable for users.

The new model selection contract is:

1. The user may explicitly select a model for the current session.
2. If the user does not select a model, the agent uses its configured model pool.
3. The planner builds one ordered list of candidates from configured, usable models.
4. Candidates are tried in order until one succeeds.
5. Auth profile failover happens inside a candidate, not as a separate top-level model routing system.

This design intentionally retires the current layered behavior where config defaults, per-agent overrides, auto-routing, session runtime state, hooks, and fallback assembly can all rewrite the active model.

## Problem Statement

Today model choice is spread across several layers:

- global default model configuration
- per-agent model overrides
- auto-routing rules that directly choose models
- session-level model overrides
- persisted runtime model fields on sessions
- hook-based provider and model overrides
- model fallback assembly in multiple places
- auth profile rotation interacting with model fallback

This creates three recurring problems:

1. It is difficult to predict which model will run for a given message.
2. Session state can unexpectedly override new config.
3. Failures are hard to diagnose because multiple systems can mutate the chosen model.

The new system should make model choice come from one place and leave session state with only the minimum information needed for explicit user control.

## Design Principles

- One planner owns model choice.
- Configured models and runtime model pools are separate concepts.
- User intent is explicit and always higher priority than automatic selection.
- Automatic selection should prefer local models first, then cloud models, and within each location try lower tiers before higher tiers.
- Session state must not silently remember the last runtime model and reuse it as future policy.
- Authentication behavior should stay inside the selected candidate, not mutate the outer model-routing order.

## Core Concepts

### Configured Models

Configured models are the source-of-truth list of models the installation knows about. A configured model is a model definition plus the metadata needed to determine whether it is eligible for runtime use.

Each configured model should carry:

- `ref`: canonical `provider/model`
- `location`: `local` or `cloud`
- `tier`: `low`, `standard`, or `high`
- `capabilities`: such as `text`, `vision`, `coding`, `tools`
- optional priority tags such as `fast` or `cheap`
- enabled/disabled state

A configured model only becomes runtime-eligible when it is fully usable:

- local provider is available and reachable
- API-key model has a usable key
- OAuth-backed model has a usable token or profile
- provider can be initialized successfully

Configured models remain visible in status and configuration views even when they are not currently runtime-eligible.

### Model Pools

Model pools are runtime selection policies over configured models. A pool does not need to hardcode every model. Instead, it describes which configured models should participate and how they should be ordered.

The default ordering rule is:

1. local before cloud
2. within the same location, `low -> standard -> high`
3. within the same tier, explicit priority if configured
4. stable tie-break ordering for predictability

Agents bind to a pool name rather than to `primary` and `fallbacks`.

The repo should support:

- a global default pool
- optional per-agent pool override

This keeps the default setup simple while still allowing specialized agents such as coding-only or vision-heavy agents to use a different pool policy.

### Session Model Selection State

Session state should be reduced to the minimum needed for explicit user control:

- `selectionMode: "auto" | "manual"`
- `manualModelRef?: string`

Session state must no longer persist runtime-chosen model values as future policy inputs.

Old runtime model fields may still exist during migration, but they should not participate in new model selection.

## Runtime Selection Flow

### Automatic Mode

When the user has not explicitly selected a model:

1. Route the message to an agent and session as usual.
2. Resolve the agent's bound model pool.
3. Build the runtime candidate list from configured models that are currently usable.
4. Apply task capability filters, such as `vision` or `coding`.
5. Order the remaining candidates according to pool policy.
6. Try candidates in order until one succeeds.
7. For each candidate, perform auth profile failover internally before moving to the next model candidate.

The planner must be the only authority that constructs this candidate list.

### Manual Mode

When the user explicitly selects a model for the current session:

1. Validate that the requested model is currently available to the agent's runtime pool view.
2. If valid, persist `selectionMode="manual"` and `manualModelRef`.
3. The planner uses that model as the session's first-choice candidate.

If the requested model is not part of the runtime-visible model set, the system should reject the request and show the available models instead of silently accepting it.

### Manual-Mode Failure Behavior

When a user-selected model fails:

1. Retry within the same model using auth/profile failover if available.
2. If that still fails, temporarily fall back within the same pool according to normal ordering rules.
3. The user-facing response should clearly state that the requested model was unavailable and that the system temporarily used another model.

This preserves user intent while keeping the system usable during transient outages.

### Reset to Automatic Mode

When the user chooses automatic mode again, for example via `/model auto`:

- clear `manualModelRef`
- set `selectionMode="auto"`
- return future turns to normal pool-based planning

## User Interface Contract

The user-facing model controls should be simplified to three actions:

- list current and available models
- set the current session model
- return the current session to automatic mode

Recommended command semantics:

- `/model` or `/model status`: show current mode, current pool, available candidates, and recent actual model usage
- `/model <provider/model>`: set manual mode for the current session
- `/model auto`: return the session to automatic mode

The system should stop exposing multiple overlapping model-selection semantics that differ by channel, session, or internal fallback layer.

## Capability Filtering

Task requirements should no longer directly choose a model. They should only filter or influence ordering inside the pool.

Examples:

- text-only chat requires `text`
- image input requires `vision`
- coding-heavy tasks require `coding` and `tools`
- background heartbeats may restrict to lower-cost tiers

This means task classification remains useful, but it no longer bypasses the pool by naming a specific model directly.

## Changes to Existing Systems

### Auto-Routing

Existing auto-routing should be retired as a direct model selector.

It may be retained only as a lightweight policy input to the planner, for example:

- start from a higher minimum tier
- require certain capabilities
- allow or disallow cloud escalation

It should no longer produce a concrete `provider/model` output.

### Hooks

Hooks should no longer be allowed to override the final chosen model. If hook behavior remains, it should only be able to add constraints or metadata before planning.

The main goal is to prevent hidden model rewrites outside the planner.

### Session Runtime Model Persistence

Persisted runtime fields such as `model` and `modelProvider` should no longer affect future selection decisions.

They may remain temporarily for:

- historical display
- usage reporting
- migration diagnostics

But they should not be read as inputs to the new planner.

### Auth Profile Rotation

Auth profile rotation remains important, but its role becomes narrower:

- it operates inside a selected model candidate
- it should not be treated as a separate top-level model routing system

This keeps provider-key failover and model failover clearly separated.

## Proposed Configuration Model

The future configuration shape should distinguish between configured models and pools.

Illustrative shape:

```json
{
  "models": {
    "catalog": {
      "ollama/qwen2.5-coder": {
        "location": "local",
        "tier": "low",
        "capabilities": ["text", "coding", "tools"]
      },
      "anthropic/claude-sonnet-4-5": {
        "location": "cloud",
        "tier": "standard",
        "capabilities": ["text", "coding", "tools", "vision"]
      },
      "openai/gpt-5.1": {
        "location": "cloud",
        "tier": "high",
        "capabilities": ["text", "coding", "tools", "vision"]
      }
    }
  },
  "agents": {
    "defaults": {
      "modelPool": "default"
    },
    "modelPools": {
      "default": {
        "locations": ["local", "cloud"],
        "tiers": ["low", "standard", "high"]
      },
      "vision": {
        "locations": ["cloud"],
        "tiers": ["standard", "high"],
        "requireCapabilities": ["vision"]
      }
    },
    "list": [
      { "id": "main", "modelPool": "default" },
      { "id": "designer", "modelPool": "vision" }
    ]
  }
}
```

The exact schema can still evolve, but the separation of concerns should remain:

- configured models describe facts
- pools describe routing policy
- agents choose a pool

## Migration Plan

### Phase 1: Introduce New Planner and Compatibility Mapping

Add the new configured-model and pool planner system while still reading old config.

Compatibility behavior:

- old `primary/fallbacks` config is mapped into a default pool at load time
- old agent model config is mapped into agent-specific pool behavior when possible
- execution still uses the new planner internally

This allows old installations to boot without preserving the old layered runtime semantics.

### Phase 2: Remove Old Runtime Model Routing

Delete or bypass the old execution-time model rewrites:

- direct auto-routing to `provider/model`
- session runtime model influence on future selection
- hook-driven model overrides
- distributed fallback list assembly in multiple layers

At this stage there should be only two runtime entry points:

- manual user selection
- automatic pool planning

### Phase 3: Clean Up Legacy Config and Session State

Remove or deprecate legacy keys from the active model-selection path:

- `models.catalog`
- `agents.defaults.modelPool`
- `agents.modelPools`
- old auto-routing model rules
- legacy session model override fields beyond the new manual-mode state

Legacy session state can be migrated as follows:

- old explicit model overrides become `selectionMode="manual"` plus `manualModelRef`
- old runtime model persistence is ignored for planning
- auth profile overrides may remain, because they are still relevant inside a selected model

## Observability and Debugging

To keep the new system supportable, logs and status output should expose the planner's decision path clearly.

Required debug information:

- routed `agentId` and `sessionKey`
- selected pool name
- configured models considered eligible
- candidates removed by capability or availability filtering
- final ordered candidate list
- chosen candidate
- whether auth failover happened inside that candidate
- whether model fallback moved to another candidate
- whether the session was in `auto` or `manual` mode

Status views should make it easy to answer:

- which pool is active
- which models are currently usable
- why a configured model is not in the runtime pool
- whether the session is pinned to a manual model
- which model actually served the most recent run

## Testing Strategy

The new system should be covered at three levels.

### Unit Tests

- eligibility filtering for configured models
- ordering rules for pools
- capability filtering behavior
- manual-mode validation and reset behavior
- auth failover remaining inside a candidate

### Integration Tests

- agent routing plus pool planning
- local-first then cloud-escalation ordering
- manual model selection persisting per session
- temporary fallback when a manual model becomes unavailable
- migration behavior from legacy config to generated pools

### Regression Tests

- session runtime model fields do not override planner decisions
- hooks cannot bypass the planner to force a final model
- auto-routing cannot directly inject `provider/model`
- fallback order remains deterministic across retries

## Recommended First Implementation Slice

The first implementation slice should focus on replacing the model decision path, not on perfect schema polish.

Recommended order:

1. Introduce configured-model eligibility resolution.
2. Introduce pool planning and candidate ordering.
3. Add session `auto/manual` selection state.
4. Switch runtime execution to planner output.
5. Move manual `/model` behavior to the new session state.
6. Retire old model-routing layers from the execution path.

This sequence gets the highest-risk complexity out first while keeping migration manageable.

## Final Decision Summary

The system will move to a single model-pool planner with these hard rules:

- user-selected model overrides automatic selection
- otherwise selection comes from the agent's model pool
- pools are built from configured, currently usable models
- local models are preferred before cloud models
- lower tiers are preferred before higher tiers
- auth failover stays inside a candidate
- session runtime model persistence no longer controls future model selection
- legacy model-routing layers are retired rather than preserved

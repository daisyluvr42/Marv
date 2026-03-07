---
title: "Auth-Backed Model Selections Design"
summary: "Let each auth method enable multiple selectable models while keeping runtime routing limited to manual override or ordered model pools."
---

# Auth-Backed Model Selections Design

## Goal

Allow one configured authentication method to enable multiple selectable models for the same provider, without reintroducing the old multi-layer runtime routing complexity.

The new contract is:

1. Authentication determines which provider family is available.
2. Marv exposes the provider's theoretical model catalog for that auth type.
3. The user selects which models they want enabled.
4. Runtime pools are built from selected models plus runtime availability state.
5. Runtime routing remains simple: `manual override` or `auto pool`, then ordered fallback.

## Problem Statement

The current pool-native model system simplified runtime routing, but it still requires users to manually maintain `models.catalog` entries for every desired model. That creates three issues:

1. One auth method often grants access to multiple models, but the config does not represent that directly.
2. Users can authenticate successfully and still see `model not allowed` because the model was never added to the pool metadata.
3. There is no clean place to remember that a selected model is permanently unsupported for the current account versus only temporarily unavailable.

We need a design where authentication, enabled models, runtime availability, and runtime routing are clearly separated.

## Design Principles

- Keep runtime routing simple.
- Move complexity into model admission, not model selection.
- Authentication and model choice are related, but not the same thing.
- Do not batch-probe models during configuration.
- Runtime validation should be lazy and safe.
- Permanent model incompatibility should remove a model from the runtime pool.
- Temporary provider problems should not permanently mutate user intent.

## Core Concepts

### Auth Profiles

`auth.profiles` continues to describe how the user authenticates with a provider, for example:

- `openai-codex` OAuth
- `google` API key
- local provider endpoint credentials

Auth profiles only answer:

- which provider family can be used
- how Marv should authenticate requests

They do not directly answer which specific models should be enabled.

### Provider Model Catalog

Marv maintains a provider-level theoretical model catalog. This is the set of models Marv knows how to address for a provider family.

This catalog does not guarantee the current account can use every model. It only defines the set of models Marv can present as selectable for that provider.

The catalog remains static or registry-driven and should not require a live probe.

### User Model Selections

Users explicitly choose which models to enable under a provider or auth profile.

This is the missing middle layer:

`auth grants -> selectable models -> user selections`

The config should represent user intent directly:

- I authenticated with Google
- I want `gemini-2.0-flash` and `gemini-2.5-flash`
- I authenticated with OpenAI Codex
- I want `gpt-5.3-codex`

### Runtime Availability State

Runtime availability is stored separately from user config and captures what the system learns while actually trying models.

Suggested statuses:

- `ready`
- `temporary_unavailable`
- `unsupported`
- `auth_invalid`

Only `unsupported` removes a model from future runtime pools automatically.

## Configuration Model

### Static Configuration

Static user intent belongs in `marv.json`.

Suggested shape:

```json
{
  "auth": {
    "profiles": {
      "openai-codex:default": {
        "provider": "openai-codex",
        "mode": "oauth"
      },
      "google:default": {
        "provider": "google",
        "mode": "api_key"
      }
    }
  },
  "models": {
    "selections": {
      "openai-codex:default": ["openai-codex/gpt-5.3-codex"],
      "google:default": ["google-gemini-cli/gemini-2.0-flash", "google-gemini-cli/gemini-2.5-flash"]
    }
  },
  "agents": {
    "defaults": {
      "modelPool": "default"
    },
    "modelPools": {
      "default": {
        "order": [
          "local/*",
          "google-gemini-cli/gemini-2.0-flash",
          "google-gemini-cli/gemini-2.5-flash",
          "openai-codex/gpt-5.3-codex"
        ]
      }
    }
  }
}
```

This makes "which models did I enable?" a first-class config concept instead of an inferred side effect of `models.catalog`.

### Runtime State Storage

Runtime-learned availability does not belong in `marv.json`.

Suggested storage:

- `~/.marv/model-availability.json`

Example:

```json
{
  "models": {
    "google-gemini-cli/gemini-2.5-flash": {
      "status": "ready",
      "lastCheckedAt": 1772880000000
    },
    "openai/gpt-5.4": {
      "status": "unsupported",
      "lastError": "model_not_available",
      "lastCheckedAt": 1772880100000
    },
    "openai-codex/gpt-5.3-codex": {
      "status": "auth_invalid",
      "lastError": "oauth_token_invalidated",
      "lastCheckedAt": 1772880200000
    }
  }
}
```

This keeps user intent and runtime facts separate.

## Runtime Pool Construction

Runtime pool construction should remain deterministic and mechanical.

For each run:

1. Resolve the active `modelPool` for the agent.
2. Collect enabled models from `models.selections`.
3. Filter out models whose provider/auth is not currently configured.
4. Filter out models marked `unsupported`.
5. Apply pool ordering rules.
6. Return the ordered candidates to the existing runtime planner.

This stage only decides which models are eligible to be tried.

## Lazy Validation

The system should not perform eager batch probing during configuration.

Instead, the first real attempt to use a selected model acts as validation.

Outcomes:

- success -> mark `ready`
- rate limit, cooldown, timeout, transient overloaded errors -> mark `temporary_unavailable`
- invalid model, no account access, deprecated model, provider says unsupported -> mark `unsupported`
- invalid or expired auth -> mark `auth_invalid`

Behavior:

- `temporary_unavailable` keeps the model in the pool
- `auth_invalid` keeps the model selected but requires reauthentication
- `unsupported` removes the model from future runtime pools until explicitly refreshed or re-enabled

This prevents repeated attempts against a model the current account can never use, without treating transient failures as permanent.

## Runtime Routing Contract

This design does not change the simplified routing path already introduced.

Runtime selection remains:

1. user manual model override for the session, or
2. automatic pool ordering

Then:

- try candidates in order
- candidate-local auth/profile failover stays inside the candidate
- move to the next candidate only when the current candidate is exhausted

No new runtime layer should:

- inject hidden model fallbacks
- rewrite the chosen model after planning
- persist the last runtime model as future policy

## Manual Selection Rules

Manual `/model <ref>` should only accept models that are:

- selectable for the configured auth/provider family
- enabled by the user
- not currently marked `unsupported`

If a model is outside that set, return an explicit "model not allowed" response and show the available choices.

When a manually selected model fails:

- retry auth/profile failover inside that model first
- then fall back within the same pool if allowed
- tell the user when the system temporarily used another model

## Error Classification

The admission/update logic needs stable categories:

- `ready`
- `temporary_unavailable`
- `unsupported`
- `auth_invalid`

Suggested mapping:

- `429`, rate limit, cooldown, overloaded, short-lived network/provider failures -> `temporary_unavailable`
- invalid model, access denied for that model, account does not support model, model removed -> `unsupported`
- invalid OAuth token, revoked token, missing key, expired auth -> `auth_invalid`

Only `unsupported` removes a model from future pools automatically.

## User Experience

Configuration and status should explain four things clearly:

1. which auth profiles are configured
2. which models are theoretically selectable for that provider
3. which models the user has enabled
4. which enabled models are currently ready, temporarily unavailable, unsupported, or auth-invalid

Recommended command behaviors:

- `/models`: browse selectable provider models and enabled selections
- `/models status`: show enabled models and runtime availability
- `/model <provider/model>`: set manual model for the session
- `/model auto`: return to automatic mode
- `/models refresh`: refresh catalog-derived selections and allow reevaluation of previously unsupported models

## Migration

The new design should migrate in phases:

1. Add `models.selections` and runtime availability storage.
2. Keep `modelPools` as the ordering mechanism.
3. Build runtime candidates from selections instead of requiring manual `models.catalog` curation.
4. Record lazy validation outcomes and remove `unsupported` models from future runtime pools.
5. Update `/models`, `/models status`, and onboarding/auth-choice flows to use the new source-of-truth.

## Success Criteria

The design is successful when:

- one auth profile can enable multiple models cleanly
- users no longer need to hand-maintain every model as pool metadata
- unsupported models are automatically removed from future runtime pools
- temporary failures do not permanently mutate user intent
- runtime routing remains limited to `manual override` or `ordered auto pool`

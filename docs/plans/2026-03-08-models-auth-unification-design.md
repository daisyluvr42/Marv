---
title: "Models Auth Unification Design"
summary: "Make `marv models auth` the single source of truth for provider auth and provider-specific model setup, while keeping `onboard` and `configure` UX stable."
---

# Models Auth Unification Design

## Goal

Make `marv models auth` the single ownership boundary for model-provider setup.

After this change:

1. Every model provider auth flow lives under `marv models auth`.
2. Provider-specific setup fields such as `apiKey`, `baseUrl`, `gatewayId`, `accountId`, and default-model recommendations are also handled there.
3. `marv onboard` and `marv configure` keep their current prompts and user-facing flow, but internally delegate model setup to `models auth`.
4. Runtime model pool inputs continue to come from `models` config, but that config is now produced through one unified model-auth layer instead of several unrelated command paths.

## Problem Statement

Today the codebase splits provider setup logic across multiple command families:

- `models auth` handles only a subset of provider flows, mostly plugin login and token helpers.
- `onboard` contains a large amount of provider-specific API key and config writing logic.
- `configure` reuses parts of onboarding logic, but still treats model auth as a wizard concern rather than a `models` concern.

That creates several problems:

1. Users cannot reliably assume "all model setup lives under `models`".
2. Provider auth behavior is duplicated across onboarding and runtime configuration flows.
3. Provider-specific config patches and auth-profile writes are not centralized, so future model-pool features risk drift.
4. The current command structure makes simple follow-up tasks awkward, for example "I already onboarded, now I just want to add a Gemini key" or "update only the Cloudflare gateway id".

## Design Principles

- `models` owns model setup.
- `onboard` and `configure` own user guidance and sequencing, not provider persistence logic.
- Provider auth and provider config should be updated atomically through one orchestration layer.
- Existing onboarding prompts should remain familiar.
- OAuth, tokens, API keys, and provider-specific metadata should use the same internal contract.
- The runtime model pool should keep reading from `models` config and auth profiles, without needing special onboarding knowledge.

## Current State

### Split ownership

The current code divides responsibility like this:

- `src/commands/models/auth.ts`
  - plugin-based `models auth login`
  - token helpers such as `setup-token` and `paste-token`
- `src/commands/auth-choice.apply.*`
  - provider-specific onboarding logic
- `src/commands/onboard-auth.credentials.ts`
  - direct auth-profile writes
- `src/commands/onboard-auth.config-*.ts`
  - provider-specific config patching
- `src/commands/configure.gateway-auth.ts`
  - wizard entry that still routes through onboarding-oriented handlers

### Practical result

Providers currently fall into three rough buckets:

- plugin-backed `models auth login` providers
- token helpers under `models auth`
- API-key or mixed-config providers that only have a complete setup path through `onboard` or `configure`

This means command ownership no longer matches the product mental model.

## Proposed Command Model

## User-facing contract

The top-level contract becomes:

- `marv models auth ...` is the canonical place to add, update, inspect, and repair model-provider auth/config.
- `marv onboard` and `marv configure` may still ask the same questions, but they call the same underlying `models auth` orchestration.

### Command families

Keep the existing commands, but expand `models auth` so it covers all providers:

- `marv models auth login ...`
  - plugin or OAuth-driven flows
- `marv models auth add ...`
  - interactive provider setup for API-key or token-based providers
- `marv models auth set ...`
  - non-interactive provider setup/update for scripting and targeted edits
- `marv models auth order ...`
  - existing per-agent auth order management

The key addition is `set`, plus a broader `add`, so every provider can be configured from `models auth`.

### Provider examples

Examples of the new target shape:

- `marv models auth add --provider google`
- `marv models auth set --provider google --api-key "$GEMINI_API_KEY"`
- `marv models auth set --provider cloudflare-ai-gateway --account-id ... --gateway-id ... --api-key ...`
- `marv models auth set --provider vllm --base-url http://localhost:8000/v1 --api-key token --model mistral-small`
- `marv models auth login --provider google-gemini-cli`

`onboard` and `configure` remain free to ask questions in a guided order, but they should translate answers into calls to the same provider operation layer.

## Internal Architecture

### Introduce a single provider setup engine

Add a new internal layer under `models auth`, conceptually:

- provider setup input
- provider setup handler
- provider setup result

Suggested home:

- `src/commands/models/auth-provider-setup.ts`
- `src/commands/models/auth-provider-handlers/*`

### Unified input contract

Each provider handler should accept a normalized input shape that can be produced by:

- direct `models auth` CLI flags
- interactive `models auth add` prompts
- onboarding prompts
- configure prompts

Suggested fields:

- `provider`
- `method`
- `profileId`
- `apiKey`
- `token`
- `baseUrl`
- `accountId`
- `gatewayId`
- `projectId`
- `model`
- `setDefault`
- `agentId`
- provider-specific optional metadata

This avoids leaking wizard-only concepts into persistence code.

### Unified result contract

Each provider handler should return a structured result that completely describes the changes to persist:

- auth profiles to upsert
- config patch to merge
- optional default model recommendation
- optional allowlist or selection hints
- user-facing notes

This generalizes what plugin `ProviderAuthResult` already does and lets API-key providers use the same model.

### Persistence path

Only the unified models-auth engine should directly perform:

- `upsertAuthProfile(...)`
- `updateConfig(...)`
- provider config patch creation
- default-model application

`onboard` and `configure` should stop writing auth or provider config themselves.

## Onboard and Configure Delegation

### Onboard

Keep:

- grouped provider selection
- provider-specific prompts
- first-run tone and recommendation flow

Change:

- `applyAuthChoice(...)` becomes an adapter that collects answers, maps them to a normalized provider setup request, and delegates to the new `models auth` engine.
- provider-specific auth-choice files become prompt mappers or thin adapters, not persistence owners.

### Configure

Keep:

- existing entry points and prompt sequence
- "configure just this part" workflow

Change:

- model-auth related configure flows also delegate to the same provider setup engine.

This preserves UX while removing duplicated write paths.

## Provider Coverage Strategy

### Fully unified providers

The new engine should cover all current model-auth styles:

- API key only
- token only
- OAuth only
- plugin login
- mixed auth + provider config

That includes providers that currently need extra metadata, such as:

- `cloudflare-ai-gateway`
- `vllm`
- `custom-api-key`
- `litellm`
- `vercel-ai-gateway`
- `qianfan`
- `moonshot`
- `minimax`

### Plugin providers

Plugin providers should stay plugin-driven for the actual auth dance, but the final persistence path should still flow through the unified result contract.

That means:

- plugin login still performs browser/device/OAuth work
- plugin result is normalized into the same persistence pipeline as API-key providers

This keeps plugin flexibility without splitting storage semantics.

## Data Ownership

### Auth profiles

`auth-profiles.json` remains the storage for provider credentials and tokens.

### Model config

`marv.json` remains the storage for:

- `auth.profiles` references
- `models.providers.*` entries
- default model recommendations
- future model-selection metadata

The change is not the storage format. The change is who is allowed to author those writes.

## Migration Plan

### Phase 1: internal unification

1. Introduce the unified provider setup engine.
2. Port existing `models auth` token/plugin logic to use it.
3. Port API-key providers from onboarding into the same engine.
4. Make `applyAuthChoice(...)` call the new engine instead of writing directly.
5. Make configure model-auth flows do the same.

### Phase 2: CLI expansion

1. Expand `models auth add` to support all providers interactively.
2. Add a non-interactive `models auth set` command for scripting and targeted updates.
3. Keep existing `login`, `setup-token`, and `paste-token` as compatibility wrappers where useful.

### Phase 3: cleanup

1. Remove direct credential writers from onboarding-oriented command paths.
2. Reduce provider-specific duplication in `auth-choice.apply.*`.
3. Consolidate docs so users are pointed to `models auth` as the durable command family.

## Compatibility

### User compatibility

- Existing `onboard` and `configure` flows should still feel the same.
- Existing auth profiles and config layout remain valid.
- Existing specialized commands should remain available unless there is a clear replacement.

### Internal compatibility

- Current provider config patch helpers can be reused at first.
- Existing plugin auth flows can keep their current interfaces and be adapted into the new result contract.

## Error Handling

The unified engine should own consistent behavior for:

- missing required provider fields
- invalid combinations of fields
- overwriting existing profiles
- partial config updates
- plugin/provider mismatch messages

Guideline:

- validate inputs before any write
- compute one combined write plan
- then persist auth profiles and config patch together in one command path

When a provider requires multiple values, error messages should mention the exact missing flags or prompt fields.

## Testing Strategy

### Unit tests

Add focused tests for:

- provider setup request normalization
- provider-specific validation
- provider-specific result generation
- config patch generation
- default-model recommendation application

### Integration tests

Add or update tests proving:

- `models auth add` can configure an API-key provider such as `google`
- `models auth set` can configure non-interactive providers with extra metadata
- `onboard` delegates to the models-auth engine instead of writing directly
- `configure` delegates to the same engine
- plugin login providers still work and persist through the unified path

### Regression tests

Keep regression coverage for:

- auth-profile file shape
- provider config shape in `marv.json`
- runtime model-pool visibility after setup
- default-model recommendation behavior

## Recommended First Slice

The safest first implementation slice is:

1. create the unified provider setup engine
2. move `google`, `openai`, `openrouter`, `xai`, `anthropic apiKey`, and one mixed-config provider into it
3. route `onboard` and `configure` through the engine for those providers
4. add `models auth set`
5. then migrate the remaining providers in groups

This reduces risk while proving the architecture on both simple and complex providers.

## Success Criteria

This work is successful when:

1. A user can configure every built-in model provider from `marv models auth`.
2. `onboard` and `configure` still work, but they no longer own provider persistence logic.
3. Auth-profile and provider-config writes have one shared implementation path.
4. Future model-pool and provider-selection work can rely on `models` as the single model-setup boundary.

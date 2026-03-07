# Auth-Backed Model Selections Implementation Plan

Date: 2026-03-07

## Summary

Implement provider-auth-backed model selections so a single auth method can enable multiple models, while preserving the current simplified runtime routing path.

The end state is:

- users authenticate once per provider or auth profile
- users choose multiple models under that auth
- runtime pools are built from selected models plus runtime availability state
- unsupported models are removed from future pools automatically
- runtime routing remains `manual override` or ordered automatic pool fallback

## Success Criteria

- A configured auth profile can back multiple selected models.
- Users no longer need to hand-maintain `models.catalog` entries just to make a model selectable.
- Runtime pool construction uses selections and availability state.
- Lazy validation marks models as `ready`, `temporary_unavailable`, `unsupported`, or `auth_invalid`.
- Only `unsupported` removes a model from future runtime pools.
- Manual `/model` accepts only enabled, currently supported models.

## Scope

### In Scope

- `models.selections` config shape
- provider/auth-backed selectable model expansion
- runtime availability state storage
- lazy validation state updates
- pool construction from selections
- `/models` and `/models status` updates
- manual model selection validation against the new enabled model set

### Out of Scope

- eager bulk probing during setup
- redesigning auth storage
- reintroducing model choice in auto-routing
- changing session routing or session-key semantics

## Phase 1: Config Shape and Types

Goal: add the new config surface for auth-backed model selections.

Tasks:

1. Add `models.selections` types and schema support.
2. Define whether selections are keyed by auth profile id, provider id, or both.
3. Keep existing pool config as the ordering layer.
4. Add validation to ensure selected model refs belong to known provider families.

Deliverable:

- config loads with `models.selections`
- status and commands can read the selections

## Phase 2: Selectable Model Expansion

Goal: derive user-selectable models from configured auth and provider catalogs without live probing.

Tasks:

1. Build a helper that maps configured auth profiles to provider families.
2. Build a helper that returns the theoretical model list for a provider family.
3. Expose a normalized "selectable models" view for commands and planners.
4. Ensure this layer does not call live provider APIs.

Deliverable:

- one place answers "what models could this auth profile select?"

## Phase 3: Runtime Availability State

Goal: keep runtime-learned model health separate from static config.

Tasks:

1. Add a persisted runtime state file for model availability.
2. Define record fields for:
   - status
   - lastCheckedAt
   - lastError
   - retryAfter when applicable
3. Add helpers to read, update, and clear entries.
4. Add reset or refresh entrypoints for users.

Deliverable:

- runtime availability state can be updated independently of `marv.json`

## Phase 4: Pool Construction From Selections

Goal: stop requiring hand-curated `models.catalog` for admission into runtime pools.

Tasks:

1. Update pool construction to start from:
   - configured auth
   - user-selected models
   - runtime availability state
2. Filter out missing-auth and `unsupported` models.
3. Preserve the existing pool ordering behavior.
4. Keep runtime planner output compatible with the current fallback runner.

Deliverable:

- runtime pools are selection-driven rather than catalog-entry-driven

## Phase 5: Lazy Validation and State Mutation

Goal: learn model availability only when a selected model is actually attempted.

Tasks:

1. Add result classification for:
   - `ready`
   - `temporary_unavailable`
   - `unsupported`
   - `auth_invalid`
2. Update model attempt/failover paths to write availability results.
3. Ensure `unsupported` excludes the model from future pools automatically.
4. Ensure `temporary_unavailable` and `auth_invalid` do not permanently remove user selections.

Deliverable:

- lazy validation updates runtime state correctly

## Phase 6: Command and UX Updates

Goal: make the new mental model visible to users.

Tasks:

1. Update `/models` to show selectable provider models and enabled selections.
2. Update `/models status` to show runtime availability states.
3. Update `/model <ref>` validation to check enabled selections plus runtime support.
4. Add `/models refresh` or equivalent recovery path for reevaluating previously unsupported models.

Deliverable:

- users can understand why a model is selectable, enabled, blocked, or removed

## Phase 7: Migration and Cleanup

Goal: remove reliance on manual per-model pool admission.

Tasks:

1. Migrate existing pool-native configs where possible.
2. Reduce `models.catalog` to an internal or compatibility role, or retire it if fully replaced.
3. Remove code paths that still treat catalog entries as the sole source of allowed model refs.
4. Add regression coverage for:
   - multi-model selection under one auth profile
   - unsupported model removal
   - temporary failure retention
   - manual model validation

Deliverable:

- the new config and runtime state are the source of truth for model admission

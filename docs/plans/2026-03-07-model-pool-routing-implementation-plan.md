# Model Pool Routing Implementation Plan

Date: 2026-03-07

## Summary

Implement the new model-pool routing system in phased slices so we can replace the current layered model-selection path without destabilizing agent execution.

The end state is:

- configured models become the source-of-truth inventory
- runtime model pools become the only automatic selection mechanism
- users can pin the current session to a manual model
- all other legacy model-rewrite layers are removed from the runtime path

## Success Criteria

- A session runs in either `auto` or `manual` model-selection mode.
- Automatic mode selects from a single ordered candidate list built from usable configured models.
- Manual mode uses the user-selected model first and falls back predictably when configured to do so.
- Runtime model persistence no longer changes future model selection.
- Auto-routing no longer emits concrete `provider/model` decisions.
- Hooks no longer override final model choice.
- `/model` surfaces the new behavior clearly.

## Scope

### In Scope

- new configured-model eligibility resolver
- new model pool planner
- new session model-selection state
- runtime integration in auto-reply execution path
- manual model selection behavior through existing `/model` surface
- migration from legacy config and session state
- logging and status visibility for pool planning
- tests for automatic selection, manual selection, and migration behavior

### Out of Scope

- redesigning provider auth storage itself
- changing channel routing or session-key semantics
- large UI redesign outside necessary model status updates
- provider-specific quality ranking beyond the new location/tier rules

## Implementation Strategy

### Phase 1: Introduce New Planning Primitives

Goal: add the new planning building blocks without changing the execution path yet.

Tasks:

1. Add a configured-model inventory module.
   - Suggested location: `src/agents/model/model-catalog-runtime.ts` or similar.
   - Responsibilities:
     - read configured model definitions
     - normalize `provider/model` refs
     - attach metadata such as `location`, `tier`, `capabilities`
     - evaluate whether a model is currently usable

2. Add a model-pool planner module.
   - Suggested location: `src/agents/model/model-pool.ts`.
   - Responsibilities:
     - resolve the active pool for an agent
     - filter configured models by pool and task requirements
     - sort candidates deterministically
     - return a structured candidate plan

3. Add new session selection-state helpers.
   - Suggested location: `src/core/session/model-selection-state.ts`.
   - Responsibilities:
     - read and write `selectionMode`
     - read and write `manualModelRef`
     - migrate legacy explicit overrides to the new state shape

4. Define the new config types.
   - Add `models.catalog` metadata shape
   - Add `agents.defaults.modelPool`
   - Add `agents.modelPools`
   - Add optional per-agent `modelPool`

Deliverable:

- planner modules and types exist
- unit tests for filtering, ordering, and session selection-state helpers

### Phase 2: Cut Runtime Over to Pool-Native Config

Goal: ensure the planner only consumes `models.catalog`, `agents.defaults.modelPool`, and `agents.modelPools`.

Tasks:

1. Remove legacy config mapping from the runtime path.
   - Automatic planning reads only pool-native config.

2. Keep session migration focused on explicit user selections.
   - The planner no longer consumes legacy default/fallback config at all.

3. Add migration helpers for existing session overrides.
   - Convert explicit old session model overrides into:
     - `selectionMode="manual"`
     - `manualModelRef`
   - Ignore old runtime `model/modelProvider` values for planning.

Deliverable:

- the runtime only consumes pool-native planner inputs

### Phase 3: Wire Automatic Pool Selection Into Execution

Goal: make automatic selection come entirely from the new planner.

Primary integration points:

- `src/auto-reply/reply/get-reply.ts`
- `src/auto-reply/reply/get-reply-run.ts`
- `src/agents/model/model-fallback.ts`
- any helper paths that currently derive provider/model before execution

Tasks:

1. Replace default model resolution in reply flow with planner entrypoint.
   - Determine:
     - active agent
     - session selection mode
     - task requirements
     - candidate list

2. Replace distributed fallback assembly with planner-owned candidate order.
   - `runWithModelFallback()` should accept an ordered candidate list from the planner instead of building its own list from multiple config sources.

3. Ensure auth failover remains inside a selected candidate.
   - Preserve provider-auth rotation logic, but make it subordinate to one chosen candidate at a time.

4. Add planner debug logging.
   - log pool name
   - log candidate eligibility filtering
   - log final candidate order
   - log chosen candidate and failover steps

Deliverable:

- automatic selection path runs entirely through the new planner

### Phase 4: Wire Manual Session Model Selection

Goal: replace current session model override semantics with the new `auto/manual` model state.

Primary integration points:

- existing `/model` command handling
- any self-settings or session-setting surfaces that can change model
- session status/reporting surfaces

Tasks:

1. Update `/model <ref>` to set manual mode.
   - validate against runtime-visible candidate set
   - persist `selectionMode="manual"` and `manualModelRef`

2. Update `/model auto` to clear manual mode.

3. Update status surfaces to show:
   - selection mode
   - manual model if present
   - active pool
   - actual most recent run model for display only

4. Update any session patch or self-settings helpers that currently write old override fields.

Deliverable:

- user model selection behaves consistently everywhere

### Phase 5: Retire Legacy Runtime Model Routing

Goal: remove the old sources of model mutation from the runtime path.

Tasks:

1. Retire direct auto-routing model selection.
   - replace with planner policy hints only, or remove entirely in first cut

2. Retire session runtime model fields as policy inputs.

3. Retire hook-based final model override behavior.

4. Retire legacy fallback assembly based on:
   - global primary/fallbacks
   - per-agent fallback merging
   - ad hoc auto-routing fallback injection

5. Remove dead helpers once planner migration is complete.

Deliverable:

- only manual mode and pool planning can change chosen model

### Phase 6: Observability and Status

Goal: make the new behavior easy to inspect and support.

Tasks:

1. Update status/reporting commands.
   - `models status`
   - session status tools
   - `/model status`

2. Surface model eligibility diagnostics.
   - configured but unusable models
   - missing auth reason class
   - provider unavailable
   - capability mismatch

3. Add concise runtime trace fields for recent choice history.

Deliverable:

- support and debugging can answer "why did this session use this model?" in one place

## Data Model Changes

### Config

Add:

- `models.catalog`
- `agents.defaults.modelPool`
- `agents.modelPools`
- optional `agents.list[].modelPool`

Deprecate from active runtime path:

- `models.catalog`
- `agents.defaults.modelPool`
- `agents.modelPools`
- direct auto-routing `rules[].model`

### Session Store

Add:

- `selectionMode?: "auto" | "manual"`
- `manualModelRef?: string`

Deprecate from planning usage:

- `model`
- `modelProvider`
- `modelOverride`
- `providerOverride`

Note:

- legacy fields may remain temporarily for migration and display, but must not influence planning once the migration is complete

## Detailed Work Items by Module

### `src/agents/model/*`

- add configured-model runtime resolver
- add pool planner
- update fallback runner to consume planner candidates
- reduce old `model-selection.ts` responsibilities to legacy compatibility and normalization helpers only

### `src/auto-reply/reply/*`

- replace scattered model-resolution calls with planner entrypoint
- replace session override reads with new selection-state helpers
- remove dependence on runtime-persisted model fields for future planning

### `src/core/session/*` and `src/core/gateway/*`

- add new session selection-state storage helpers
- update session status/reporting to expose new state
- keep actual last-run model as informational only

### `src/commands/models/*` and related command surfaces

- update status/list/set behavior to reflect pools and session mode
- add validation against runtime-visible pool candidates
- keep legacy commands working where feasible, but route behavior through new planner/state

### `src/agents/auto-routing.ts`

- either remove direct model outputs entirely
- or narrow output to policy hints such as required capabilities or minimum tier

### Hook Integration

- remove final `provider/model` override support from runtime hooks
- if needed, allow only planner-safe hints

## Test Plan

### Unit Tests

- configured model eligibility resolution
- pool filtering by location/tier/capabilities
- candidate ordering
- manual session state read/write
- migration helpers for legacy config and session state

### Integration Tests

- auto mode chooses local low before local standard before cloud
- cloud escalation happens only after local candidates fail or are ineligible
- manual mode pins the chosen model for the session
- manual mode falls back predictably when allowed
- old config maps into the same candidate order as expected

### Regression Tests

- persisted runtime model does not override planner output
- auto-routing no longer injects concrete `provider/model`
- hooks cannot force a final model
- fallback order remains stable across retries
- auth profile failover stays within a candidate before advancing to the next candidate

## Rollout Order

Recommended merge sequence:

1. planner modules and config types
2. compatibility mapper from legacy config
3. execution-path integration for automatic mode
4. manual session mode and `/model` integration
5. observability updates
6. legacy runtime-path removals and cleanup

This order reduces risk by letting us land the new primitives before deleting old behavior.

## Risks and Mitigations

### Risk: legacy installs change behavior unexpectedly

Mitigation:

- keep config compatibility mapping explicit and tested
- add planner debug output so differences are explainable

### Risk: manual selection semantics confuse users during migration

Mitigation:

- make `/model status` show `auto` vs `manual`
- make `/model auto` the single reset path

### Risk: local models are configured but often unavailable

Mitigation:

- treat usability separately from configuration
- make ineligible reasons visible in status and logs

### Risk: runtime paths still accidentally read old session fields

Mitigation:

- add regression tests that seed legacy fields and verify planner output is unchanged
- centralize session model-state access behind one helper

## Recommended First PR Slice

The first PR should avoid command-surface churn and focus on the planning core.

Suggested first PR:

1. add configured-model eligibility resolver
2. add pool planner
3. add compatibility mapper from current config
4. add unit tests for ordering and filtering
5. add no-op wiring hooks or debug command output if helpful

This gives a reviewable core before touching the reply path.

## Follow Up After First PR

Second PR:

- integrate planner into automatic reply execution
- route fallback execution through planner candidates

Third PR:

- add manual session mode and `/model` updates
- migrate old explicit session overrides

Fourth PR:

- remove old runtime rewrites
- simplify auto-routing
- simplify hook behavior

## Final Deliverable Shape

At the end of implementation:

- one planner owns model choice
- configured models feed runtime pools
- users can pin or unpin the current session
- local-first, tier-ordered automatic selection is deterministic
- old layered model-routing logic is gone from the active path

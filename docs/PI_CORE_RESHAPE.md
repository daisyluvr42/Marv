# PI-Core Reshape Plan (Marv Hybrid)

This project is being reshaped around a pi-mono style core while retaining Marv strengths:

- multi-channel ingress and orchestration
- approval and grant model
- session workspace isolation
- scheduler and heartbeat runtime
- memory retrieval and compaction pipeline

## Implemented in this phase

1. `backend/pi_core/` compatibility layer
- unified turn context model (`PiTurnContext`, `PiMessage`)
- context transform utilities
- OpenAI-compatible message conversion

2. Processor integration
- task execution now builds a pi-compatible turn context first
- existing memory, tool loop, routing, approvals, and runtime behavior remain intact

3. Provider capability surface
- `/v1/system/core/capabilities`
- `/v1/system/core/models`
- enables provider/model discovery needed for pi-style model routing and provider governance
- provider auth layering with per-provider env lookup:
  - api key: `CORE_PROVIDER_<NAME>_API_KEY`
  - oauth token: `CORE_PROVIDER_<NAME>_OAUTH_TOKEN`
  - optional explicit env override per provider matrix entry

4. Turn-level structured observability
- new `PiTurnEvent` in ledger timeline (`stage=context_ready`)
- keeps audit trail aligned with pi-style turn processing

5. Package contract layer (pi-style packages)
- package root scan (`EDGE_PACKAGES_ROOT`, default `./packages`)
- manifest contract (`MARV_PACKAGE.json` or `package.json.marvPackage`)
- runtime hook reload endpoint and CLI

## Why this keeps your stronger architecture

- Approvals and risk gating remain the source of truth.
- Session workspace constraints remain unchanged.
- Existing IM/scheduler/skills modules do not break.
- Core message flow now has a clean migration seam for future pi-style engines.

## Next phases

1. Provider core parity
- model catalog/discovery endpoint and richer provider capability metadata
- oauth/api token strategy per provider

2. Agent-core parity
- structured turn events beyond current ledger events
- tool execution abstraction that can support alternate engines

3. Package ecosystem parity
- plugin/package contracts similar to pi-mono packages
- optional bridges for Slack manager and pod runtimes

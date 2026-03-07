# Gateway Config Write Manual

Use this guide whenever Marv is changing its own config through the gateway tool or RPC.

## Core rule

- Do not hand-edit the active config file with shell tools, `write`, `edit`, or `apply_patch`
- Use the gateway tool with `action: "config.get"` first
- Then use exactly one of:
  - `config.patch` for partial updates
  - `config.apply` for full replacement
  - `config.patches.propose` / `config.patches.commit` for semantic patch flow

## Required write sequence

1. Call `gateway` with `action: "config.get"`
2. Treat `result.activeConfigPath` as authoritative; fall back to `result.path`
3. Capture `result.hash` and pass it back as `baseHash` on the write call
4. Perform the write
5. If the write is rejected because config changed, call `config.get` again and retry from the new snapshot

If config already exists and `baseHash` is missing or stale, the write will be rejected.

## When to use each write method

### `config.patch`

Use for partial changes.

Behavior:

- `raw` must be a JSON5 object containing only the keys to change
- Objects merge recursively
- `null` deletes a key
- Arrays usually replace
- Some arrays of objects keyed by `id` merge by `id`

Important:

- `config.patch` can create missing keys, including new top-level branches
- But the merged result must still pass validation as a complete config
- If you create a new object branch, any required fields for that branch still need to be present in the patch

Do not assume "patch can only modify existing keys". That is false.

### `config.apply`

Use only when replacing the entire config.

Behavior:

- `raw` must be the full JSON5 config object
- Anything omitted is removed
- Validation runs against the full submitted object before write

Do not use `config.apply` with a partial object. If you only want to change one section, use `config.patch`.

## Redaction and sensitive values

`config.get` may return redacted values such as `__MARV_REDACTED__`.

Rules:

- Do not treat `__MARV_REDACTED__` as a real credential
- It is safe only as a placeholder for an already-existing sensitive field from the current config snapshot
- Do not copy that sentinel into a newly created sensitive field or a field that did not exist in the original config
- If setting a new secret, provide the real value or an env placeholder such as `${ENV_VAR}`

## Failure modes to recognize

### "config base hash required" or "config changed since last load"

- Cause: missing or stale `baseHash`
- Fix: call `config.get` again and retry with the latest `hash`

### "invalid config"

- Cause: the resulting full config failed schema validation
- Fix: inspect the returned issues; do not assume the patch mechanism itself is broken

Common examples:

- Created a new branch but omitted fields that the branch requires
- Added an unknown top-level key
- Replaced a full object with an incomplete one

### Redaction sentinel errors

- Cause: `__MARV_REDACTED__` appeared in a place that cannot be restored from the original config
- Fix: replace it with the real secret, an env placeholder, or leave the existing field untouched

## Workspace rule

Do not rely on repository docs such as `docs/gateway/configuration.md` being visible from the runtime workspace. This bundled reference is the source of truth for gateway config writes when running from `Marv-Run` or another reduced workspace.

## Safe defaults

- Prefer `marv config set` for a single known key
- Prefer `config.patch` for targeted gateway-driven edits
- Prefer `config.apply` only after confirming the payload is a complete config snapshot
- When the current config is invalid, stop and report it or run `marv doctor`; do not rewrite from scratch unless explicitly asked

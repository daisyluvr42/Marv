---
summary: "CLI reference for `marv models` (status/list/set/scan, aliases, fallbacks, pool, auth)"
read_when:
  - You want to change default models or view provider auth status
  - You want to scan available models/providers and debug auth profiles
  - You want to manage runtime model availability (pool list/clear)
title: "models"
---

# `marv models`

Model discovery, scanning, and configuration (default model, fallbacks, auth profiles).

Related:

- Providers + models: [Models](/providers/models)
- Provider auth setup: [Getting started](/start/getting-started)

## Common commands

```bash
marv models status
marv models list
marv models set <model-or-alias>
marv models scan
```

`marv models status` shows the resolved default/fallbacks plus an auth overview.
When provider usage snapshots are available, the OAuth/token status section includes
provider usage headers.
Add `--probe` to run live auth probes against each configured provider profile.
Probes are real requests (may consume tokens and trigger rate limits).
Use `--agent <id>` to inspect a configured agent’s model/auth state. When omitted,
the command uses `MARV_AGENT_DIR`/`PI_CODING_AGENT_DIR` if set, otherwise the
configured default agent.

Notes:

- `models set <model-or-alias>` accepts `provider/model` or an alias.
- Model refs are parsed by splitting on the **first** `/`. If the model ID includes `/` (OpenRouter-style), include the provider prefix (example: `openrouter/moonshotai/kimi-k2`).
- If you omit the provider, Marv treats the input as an alias or a model for the **default provider** (only works when there is no `/` in the model ID).

### `models status`

Options:

- `--json`
- `--plain`
- `--check` (exit 1=expired/missing, 2=expiring)
- `--probe` (live probe of configured auth profiles)
- `--probe-provider <name>` (probe one provider)
- `--probe-profile <id>` (repeat or comma-separated profile ids)
- `--probe-timeout <ms>`
- `--probe-concurrency <n>`
- `--probe-max-tokens <n>`
- `--agent <id>` (configured agent id; overrides `MARV_AGENT_DIR`/`PI_CODING_AGENT_DIR`)

## Pool management

```bash
marv models pool list              # show runtime availability state
marv models pool list --json       # JSON output
marv models pool clear             # clear all availability entries
marv models pool clear <model-ref> # clear a specific model
```

`models pool list` shows the runtime availability state for each model: status
(`ready`, `temporary_unavailable`, `unsupported`, `auth_invalid`), last check
time, retry countdown, and last error.

`models pool clear` removes failure entries so models re-enter the candidate
pool immediately. This is useful when a local model was transiently unavailable
(slow cold-start, server restart) and got marked as failed.

## Aliases + fallbacks

```bash
marv models aliases list
marv models fallbacks list
```

## Auth profiles

```bash
marv models auth add
marv models auth add --provider google --set-default
marv models auth set --provider google --method gemini-api-key --api-key "$GEMINI_API_KEY"
marv models auth login --provider <id>
marv models auth setup-token
marv models auth paste-token
```

`models auth add` is the general interactive setup entrypoint for model providers.
Use it to configure API key providers and provider-specific settings without
re-running full onboarding.

`models auth set` is the non-interactive version for scripts and targeted updates.
Use it when you want to set API keys or provider-specific fields directly from the
CLI without prompts.

`models auth login` runs a provider plugin’s auth flow (OAuth/API key). Use
`marv plugins list` to see which providers are installed.

Notes:

- `add` accepts `--provider`, `--method`, and `--set-default`.
- `set` accepts `--provider`, optional `--method`, and provider-specific flags such as `--api-key`, `--token`, `--base-url`, `--model`, `--provider-id`, `--account-id`, `--gateway-id`, and `--set-default`.
- `setup-token` prompts for a setup-token value (generate it with `claude setup-token` on any machine).
- `paste-token` accepts a token string generated elsewhere or from automation.

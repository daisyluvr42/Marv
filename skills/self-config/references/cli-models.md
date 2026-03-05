# Models CLI Reference

## marv models list

List models (configured by default, or all available):

```bash
marv models list
marv models list --all
```

## marv models status

Show configured model state:

```bash
marv models status
```

## marv models set

Set the default model:

```bash
marv models set <model-id>
```

## marv models set-image

Set the image generation model:

```bash
marv models set-image <model-id>
```

## marv models scan

Scan provider catalogs for available models:

```bash
marv models scan
```

## Model aliases

```bash
marv models aliases list             # list aliases
marv models aliases add <alias> <model>  # add/update alias
marv models aliases remove <alias>   # remove alias
```

## Fallback models

Fallback chain used when the primary model fails:

```bash
marv models fallbacks list           # list fallbacks
marv models fallbacks add <model>    # add fallback
marv models fallbacks remove <model> # remove fallback
marv models fallbacks clear          # clear all
```

Image fallbacks:

```bash
marv models image-fallbacks list
marv models image-fallbacks add <model>
marv models image-fallbacks remove <model>
marv models image-fallbacks clear
```

## Model auth

Interactive auth helpers for model providers:

```bash
marv models auth add                          # interactive auth wizard
marv models auth login                        # run provider plugin auth flow
marv models auth setup-token                  # run provider CLI to create/sync token
marv models auth paste-token                  # paste token into auth-profiles.json
marv models auth login-github-copilot         # GitHub Copilot device flow
```

Auth order overrides (per agent):

```bash
marv models auth order get                    # show per-agent auth order
marv models auth order set <providers...>     # set auth order
marv models auth order clear                  # clear override
```

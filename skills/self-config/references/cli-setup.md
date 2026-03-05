# Setup & Maintenance CLI Reference

## marv setup

Initialize local config and agent workspace:

```bash
marv setup
```

Creates `~/.marv/marv.json` if missing, sets up default workspace.

## marv onboard

Interactive onboarding wizard for gateway, workspace, and skills:

```bash
marv onboard
marv onboard --install-daemon        # also install as background service
```

Walks through: model provider setup, channel configuration, workspace creation, skill installation.

## marv update

Update Marv:

```bash
marv update                          # interactive or auto update
marv update wizard                   # interactive update wizard
marv update status                   # show update channel and version info
```

## marv reset

Reset local config and state (CLI stays installed):

```bash
marv reset
```

Removes `~/.marv/marv.json` and related state. Does not uninstall the CLI binary.

## marv uninstall

Uninstall the gateway service and local data (CLI remains):

```bash
marv uninstall
```

## marv completion

Generate shell completion scripts:

```bash
marv completion bash                 # bash completion
marv completion zsh                  # zsh completion
marv completion fish                 # fish completion
```

Add to your shell profile:

```bash
# bash
eval "$(marv completion bash)"

# zsh
eval "$(marv completion zsh)"
```

## Installation methods

### npm (recommended)

```bash
npm install -g marv
```

### From source

```bash
git clone https://github.com/daisyluvr42/Marv.git
cd Marv && pnpm install
pnpm marv onboard --install-daemon
```

### Verify installation

```bash
marv --version
marv health
marv status --all
```

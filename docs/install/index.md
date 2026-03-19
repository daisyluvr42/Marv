---
summary: "Install Marv — installer script, npm/pnpm, from source, Docker, and more"
read_when:
  - You need an install method other than the Getting Started quickstart
  - You want to deploy to a cloud platform
  - You need to update, migrate, or uninstall
title: "Install"
---

# Install

Already followed [Getting Started](/start/getting-started)? You're all set — this page is for alternative install methods, platform-specific instructions, and maintenance.

## System requirements

- **[Node 22+](/install/node)** (the [installer script](#install-methods) will install it if missing)
- macOS, Linux, or Windows
- `pnpm` only if you build from source

<Note>
On Windows, we strongly recommend running Marv under [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install).
</Note>

## Install methods

<Tip>
The **installer script** is the recommended way to install Marv. It handles Node detection, installation, and onboarding in one step.
</Tip>

<Warning>
For VPS/cloud hosts, avoid third-party "1-click" marketplace images when possible. Prefer a clean base OS image (for example Ubuntu LTS), then install Marv yourself with the installer script.
</Warning>

<AccordionGroup>
  <Accordion title="Installer script" icon="rocket" defaultOpen>
    Downloads the CLI, installs it globally via npm, and launches the onboarding wizard.

    <Tabs>
      <Tab title="macOS / Linux / WSL2">
        ```bash
        curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash
        ```
      </Tab>
      <Tab title="Windows (PowerShell)">
        ```powershell
        iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1 | iex
        ```
      </Tab>
    </Tabs>

    That's it — the script handles Node detection, installation, and onboarding.

    To skip onboarding and just install the binary:

    <Tabs>
      <Tab title="macOS / Linux / WSL2">
        ```bash
        curl -fsSL https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash -s -- --no-onboard
        ```
      </Tab>
      <Tab title="Windows (PowerShell)">
        ```powershell
        & ([scriptblock]::Create((iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1))) -NoOnboard
        ```
      </Tab>
    </Tabs>

    For all flags, env vars, and CI/automation options, see [Installer internals](/install/installer).

  </Accordion>

  <Accordion title="npm / pnpm" icon="package">
    If you already have Node 22+ and want the simplest machine-local install, use a global package install. This is the best choice for most servers, remote machines, and single-user deployments.

    <Tip>
    Choose this path if you want to run Marv, not modify Marv. You do not need the Git checkout, TypeScript source tree, or repo devDependencies.
    </Tip>

    <Tabs>
      <Tab title="npm">
        ```bash
        npm install -g agentmarv@latest
        marv onboard --install-daemon
        ```

        Verify the install:

        ```bash
        marv gateway status
        marv dashboard
        marv tui
        ```

        Run in the foreground instead of a service:

        ```bash
        marv gateway run --bind loopback --port 4242
        ```

        <Accordion title="sharp build errors?">
          If you have libvips installed globally (common on macOS via Homebrew) and `sharp` fails, force prebuilt binaries:

          ```bash
          SHARP_IGNORE_GLOBAL_LIBVIPS=1 npm install -g agentmarv@latest
          ```

          If you see `sharp: Please add node-gyp to your dependencies`, either install build tooling (macOS: Xcode CLT + `npm install -g node-gyp`) or use the env var above.
        </Accordion>
      </Tab>
      <Tab title="pnpm">
        ```bash
        pnpm add -g agentmarv@latest
        pnpm approve-builds -g        # approve marv, node-llama-cpp, sharp, etc.
        marv onboard --install-daemon
        ```

        Verify the install:

        ```bash
        marv gateway status
        marv dashboard
        marv tui
        ```

        <Note>
        pnpm requires explicit approval for packages with build scripts. After the first install shows the "Ignored build scripts" warning, run `pnpm approve-builds -g` and select the listed packages.
        </Note>
      </Tab>
    </Tabs>

    Common fit for global installs:

    - VPS or cloud machine running a single Marv instance
    - Local machine where you want upgrades to stay simple
    - Deployments where you do not want source-only tooling such as `tsx` or UI build dependencies

    Common update path:

    ```bash
    npm install -g agentmarv@latest
    ```

  </Accordion>

  <Accordion title="From source" icon="github">
    For contributors, local development, debugging, or anyone who needs to run from a Git checkout.

    <Warning>
    A source checkout is not the same as a global install. Do not run `npm install` in the repo root. Use `pnpm install` so workspace packages, `tsx`, and the UI build pipeline are installed correctly.
    </Warning>

    <Steps>
      <Step title="Clone the repo">
        Clone the [Marv repo](https://github.com/daisyluvr42/Marv) and enter the checkout:

        ```bash
        git clone https://github.com/daisyluvr42/Marv.git
        cd Marv
        ```
      </Step>
      <Step title="Install workspace dependencies">
        Install the full workspace with pnpm:

        ```bash
        pnpm install
        ```
      </Step>
      <Step title="Build the CLI and Control UI assets">
        Build both the TypeScript output and the browser UI:

        ```bash
        pnpm ui:build
        pnpm build
        ```
      </Step>
      <Step title="Choose how you want to run commands">
        Option A: stay inside the repo and use the workspace command:

        ```bash
        pnpm marv gateway run --bind loopback --port 4242
        pnpm marv tui
        ```

        Option B: make `marv` available globally from the checkout:

        ```bash
        pnpm link --global
        ```

        Then run:

        ```bash
        marv onboard --install-daemon
        marv gateway status
        marv tui
        ```
      </Step>
      <Step title="Run onboarding">
        If you did not link globally, run onboarding from the repo:

        ```bash
        pnpm marv onboard --install-daemon
        ```

        If you linked globally, this works too:

        ```bash
        marv onboard --install-daemon
        ```
      </Step>
    </Steps>

    Source-checkout troubleshooting:

    - If you see `Cannot find package 'tsx'`, rerun `pnpm install`.
    - If you see `Cannot find package '@homebridge/ciao'`, rerun `pnpm install`.
    - If you see `Control UI assets missing` or `Control UI build failed`, run `pnpm ui:build` and then `pnpm build`.
    - If you accidentally ran `npm install` in the repo root, remove `node_modules` and `package-lock.json`, then reinstall with `pnpm install`.

    Fast repair sequence:

    ```bash
    rm -rf node_modules package-lock.json
    corepack enable
    pnpm install
    pnpm ui:build
    pnpm build
    pnpm marv gateway run --bind loopback --port 4242
    ```

    For deeper development workflows, see [Setup](/start/setup).

  </Accordion>
</AccordionGroup>

## Other install methods

<CardGroup cols={2}>
  <Card title="Docker" href="/install/docker" icon="container">
    Containerized or headless deployments.
  </Card>
  <Card title="Podman" href="/install/podman" icon="container">
    Rootless container: run `setup-podman.sh` once, then the launch script.
  </Card>
  <Card title="Nix" href="/install/nix" icon="snowflake">
    Declarative install via Nix.
  </Card>
  <Card title="Ansible" href="/install/ansible" icon="server">
    Automated fleet provisioning.
  </Card>
  <Card title="Bun" href="/install/bun" icon="zap">
    CLI-only usage via the Bun runtime.
  </Card>
</CardGroup>

## macOS companion app

The macOS menu-bar companion app (notifications, permissions, Canvas, Screen Recording) must be built from source. See [macOS App](/platforms/macos#installing-the-macos-app) for instructions.

## After install

Verify everything is working:

```bash
marv doctor         # check for config issues
marv status         # gateway status
marv dashboard      # open the browser UI
```

If you need custom runtime paths, use:

- `MARV_HOME` for home-directory based internal paths
- `MARV_STATE_DIR` for mutable state location
- `MARV_CONFIG_PATH` for config file location

See [Environment vars](/help/environment) for precedence and full details.

## Troubleshooting: `marv` not found

<Accordion title="PATH diagnosis and fix">
  Quick diagnosis:

```bash
node -v
npm -v
npm prefix -g
echo "$PATH"
```

If `$(npm prefix -g)/bin` (macOS/Linux) or `$(npm prefix -g)` (Windows) is **not** in your `$PATH`, your shell can't find global npm binaries (including `marv`).

Fix — add it to your shell startup file (`~/.zshrc` or `~/.bashrc`):

```bash
export PATH="$(npm prefix -g)/bin:$PATH"
```

On Windows, add the output of `npm prefix -g` to your PATH.

Then open a new terminal (or `rehash` in zsh / `hash -r` in bash).
</Accordion>

## Update / uninstall

<CardGroup cols={3}>
  <Card title="Updating" href="/install/updating" icon="refresh-cw">
    Keep Marv up to date.
  </Card>
  <Card title="Migrating" href="/install/migrating" icon="arrow-right">
    Move to a new machine.
  </Card>
  <Card title="Uninstall" href="/install/uninstall" icon="trash-2">
    Remove Marv completely.
  </Card>
</CardGroup>

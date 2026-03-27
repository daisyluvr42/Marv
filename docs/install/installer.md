---
summary: "How the installer scripts work (install.sh, install-cli.sh, install.ps1), flags, and automation"
read_when:
  - You want to understand the installer scripts hosted on GitHub
  - You want to automate installs (CI / headless)
  - You want to install from a GitHub checkout
title: "Installer Internals"
---

# Installer internals

Marv ships three installer scripts, served from [GitHub](https://github.com/daisyluvr42/Marv/tree/main/install).

| Script                             | Platform             | What it does                                                                             |
| ---------------------------------- | -------------------- | ---------------------------------------------------------------------------------------- |
| [`install.sh`](#installsh)         | macOS / Linux / WSL  | Installs Node if needed, installs Marv via npm (default) or git, and can run onboarding. |
| [`install-cli.sh`](#install-clish) | macOS / Linux / WSL  | Installs Node + Marv into a local prefix (`~/.marv`). No root required.                  |
| [`install.ps1`](#installps1)       | Windows (PowerShell) | Installs Node if needed, installs Marv via npm (default) or git, and can run onboarding. |

## Quick commands

<Tabs>
  <Tab title="install.sh">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash
    ```

    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash -s -- --help
    ```

  </Tab>
  <Tab title="install-cli.sh">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install-cli.sh | bash
    ```

    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install-cli.sh | bash -s -- --help
    ```

  </Tab>
  <Tab title="install.ps1">
    ```powershell
    iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1 | iex
    ```

    ```powershell
    & ([scriptblock]::Create((iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1))) -Tag beta -NoOnboard -DryRun
    ```

  </Tab>
</Tabs>

<Note>
If install succeeds but `marv` is not found in a new terminal, see [Node.js troubleshooting](/install/node#troubleshooting).
</Note>

---

## install.sh

<Tip>
Recommended for most interactive installs on macOS/Linux/WSL.
</Tip>

### Flow (install.sh)

<Steps>
  <Step title="Detect OS">
    Supports macOS and Linux (including WSL). If macOS is detected, installs Homebrew if missing.
  </Step>
  <Step title="Ensure Node.js 22+">
    Checks Node version and installs Node 22 if needed (Homebrew on macOS, NodeSource setup scripts on Linux apt/dnf/yum).
  </Step>
  <Step title="Ensure Git">
    Installs Git if missing.
  </Step>
  <Step title="Install Marv">
    - `npm` method (default): global npm install
    - `git` method: global npm install from a Git URL (`git+<repo>#<ref>`)
  </Step>
  <Step title="Post-install tasks">
    - Attempts onboarding when appropriate (onboarding not disabled)
    - On macOS, prefers downloading and launching the matching Marv app when onboarding is enabled
    - Defaults `SHARP_IGNORE_GLOBAL_LIBVIPS=1`
  </Step>
</Steps>

### Behavior notes

- `--install-method git` does **not** create or update a working checkout. It tells npm to install from the selected Git URL and ref.
- `--no-git-update` is accepted for compatibility, but the current script only warns when it is passed outside `--install-method git`.
- Unknown flags cause an immediate error.

### Examples (install.sh)

<Tabs>
  <Tab title="Default">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash
    ```
  </Tab>
  <Tab title="Skip onboarding">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash -s -- --no-onboard
    ```
  </Tab>
  <Tab title="Git install">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash -s -- --install-method git
    ```
  </Tab>
  <Tab title="Dry run">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash -s -- --dry-run
    ```
  </Tab>
</Tabs>

<AccordionGroup>
  <Accordion title="Flags reference">

| Flag                            | Description                                               |
| ------------------------------- | --------------------------------------------------------- |
| `--install-method npm\|git`     | Choose install method (default: `npm`). Alias: `--method` |
| `--version <version\|dist-tag>` | npm version or dist-tag (default: `latest`)               |
| `--beta`                        | Use beta dist-tag if available, else fallback to `latest` |
| `--package <path-or-url>`       | Install from a local or remote package tarball            |
| `--repo <url>`                  | Git repo URL used with `--install-method git`             |
| `--ref <ref>`                   | Git ref used with `--install-method git`                  |
| `--no-git-update`               | Compatibility flag; ignored by npm installs               |
| `--set-npm-prefix`              | Force npm global prefix to `~/.npm-global` when needed    |
| `--no-onboard`                  | Skip onboarding                                           |
| `--onboard`                     | Enable onboarding                                         |
| `--no-mac-app`                  | Skip Marv app download on macOS and use CLI onboarding    |
| `--dry-run`                     | Print actions without applying changes                    |
| `--verbose`                     | Enable shell tracing                                      |
| `--help`                        | Show usage (`-h`)                                         |

  </Accordion>

  <Accordion title="Environment variables reference">

| Variable                            | Description                                    |
| ----------------------------------- | ---------------------------------------------- |
| `MARV_INSTALL_METHOD=git\|npm`      | Install method                                 |
| `MARV_VERSION=<version-or-disttag>` | npm version or dist-tag                        |
| `MARV_PACKAGE=<path-or-url>`        | Install from a local or remote package tarball |
| `MARV_REPO=<url>`                   | Git repo URL                                   |
| `MARV_REF=<ref>`                    | Git ref                                        |
| `MARV_NO_ONBOARD=1`                 | Skip onboarding                                |
| `MARV_NO_MAC_APP=1`                 | Skip Marv app download on macOS                |
| `MARV_DRY_RUN=1`                    | Dry run mode                                   |
| `MARV_VERBOSE=1`                    | Debug mode                                     |
| `SHARP_IGNORE_GLOBAL_LIBVIPS=0\|1`  | Control sharp/libvips behavior (default: `1`)  |

  </Accordion>
</AccordionGroup>

---

## install-cli.sh

<Info>
Designed for environments where you want everything under a local prefix (default `~/.marv`) and no system Node dependency.
</Info>

### Flow (install-cli.sh)

<Steps>
  <Step title="Install local Node runtime">
    Downloads Node tarball (default `22.22.0`) to `<prefix>/tools/node-v<version>` and verifies SHA-256.
  </Step>
  <Step title="Prepare local wrapper">
    Creates `<prefix>/package.json`, `<prefix>/home`, and the `<prefix>/bin/marv` wrapper.
  </Step>
  <Step title="Install Marv under prefix">
    Installs with npm using `--prefix <prefix>`, then writes wrapper to `<prefix>/bin/marv`.
  </Step>
</Steps>

### Examples (install-cli.sh)

<Tabs>
  <Tab title="Default">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install-cli.sh | bash
    ```
  </Tab>
  <Tab title="Custom prefix + version">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install-cli.sh | bash -s -- --prefix /opt/marv --version latest
    ```
  </Tab>
  <Tab title="Automation JSON output">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install-cli.sh | bash -s -- --json --prefix /opt/marv
    ```
  </Tab>
  <Tab title="Run onboarding">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install-cli.sh | bash -s -- --onboard
    ```
  </Tab>
</Tabs>

<AccordionGroup>
  <Accordion title="Flags reference">

| Flag                   | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `--prefix <path>`      | Install prefix (default: `~/.marv`)                         |
| `--version <ver>`      | Marv version or dist-tag (default: `latest`)                |
| `--node-version <ver>` | Node version (default: `22.22.0`)                           |
| `--json`               | Emit NDJSON events                                          |
| `--onboard`            | Run `marv onboard` after install                            |
| `--no-onboard`         | Skip onboarding (default)                                   |
| `--set-npm-prefix`     | Accepted for compatibility; local-prefix installs ignore it |
| `--help`               | Show usage (`-h`)                                           |

  </Accordion>

  <Accordion title="Environment variables reference">

| Variable                           | Description                                    |
| ---------------------------------- | ---------------------------------------------- |
| `MARV_PREFIX=<path>`               | Install prefix                                 |
| `MARV_VERSION=<ver>`               | Marv version or dist-tag                       |
| `MARV_NODE_VERSION=<ver>`          | Node version                                   |
| `MARV_PACKAGE=<path-or-url>`       | Install from a local or remote package tarball |
| `MARV_NO_ONBOARD=1`                | Skip onboarding                                |
| `SHARP_IGNORE_GLOBAL_LIBVIPS=0\|1` | Control sharp/libvips behavior (default: `1`)  |

  </Accordion>
</AccordionGroup>

---

## install.ps1

### Flow (install.ps1)

<Steps>
  <Step title="Ensure PowerShell + Windows environment">
    Requires PowerShell 5+.
  </Step>
  <Step title="Ensure Node.js 22+">
    If missing, attempts install via winget, then Chocolatey, then Scoop.
  </Step>
  <Step title="Install Marv">
    - `npm` method (default): global npm install using the selected `-Version`
    - `git` method: global npm install from a Git URL (`git+<repo>#<ref>`)
  </Step>
  <Step title="Post-install tasks">
    Resolves the installed `marv` command, prints its version, and optionally runs onboarding.
  </Step>
</Steps>

### Examples (install.ps1)

<Tabs>
  <Tab title="Default">
    ```powershell
    iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1 | iex
    ```
  </Tab>
  <Tab title="Git install">
    ```powershell
    & ([scriptblock]::Create((iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1))) -InstallMethod git
    ```
  </Tab>
  <Tab title="Dry run">
    ```powershell
    & ([scriptblock]::Create((iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1))) -DryRun
    ```
  </Tab>
  <Tab title="Debug trace">
    ```powershell
    # install.ps1 has no dedicated -Verbose flag yet.
    Set-PSDebug -Trace 1
    & ([scriptblock]::Create((iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1))) -NoOnboard
    Set-PSDebug -Trace 0
    ```
  </Tab>
</Tabs>

<AccordionGroup>
  <Accordion title="Flags reference">

| Flag                      | Description                                 |
| ------------------------- | ------------------------------------------- |
| `-InstallMethod npm\|git` | Install method (default: `npm`)             |
| `-Version <tag>`          | npm dist-tag or version (default: `latest`) |
| `-Package <path-or-url>`  | Install from a local or remote package file |
| `-Repo <url>`             | Git repo URL for `-InstallMethod git`       |
| `-Ref <git-ref>`          | Git ref for `-InstallMethod git`            |
| `-NoOnboard`              | Skip onboarding                             |
| `-Onboard`                | Force onboarding after install              |
| `-Beta`                   | Shortcut for `-Version beta`                |
| `-DryRun`                 | Print actions only                          |

  </Accordion>

  <Accordion title="Environment variables reference">

| Variable                       | Description                 |
| ------------------------------ | --------------------------- |
| `MARV_INSTALL_METHOD=git\|npm` | Install method              |
| `MARV_VERSION=<version>`       | npm dist-tag or version     |
| `MARV_PACKAGE=<path-or-url>`   | Package tarball path or URL |
| `MARV_REPO=<url>`              | Git repo URL                |
| `MARV_REF=<ref>`               | Git ref                     |
| `MARV_NO_ONBOARD=1`            | Skip onboarding             |
| `MARV_DRY_RUN=1`               | Dry run mode                |

  </Accordion>
</AccordionGroup>

<Note>
If `-InstallMethod git` is used and Git is unavailable, npm will fail while resolving the Git dependency.
</Note>

---

## CI and automation

Use non-interactive flags/env vars for predictable runs.

<Tabs>
  <Tab title="install.sh (non-interactive npm)">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash -s -- --no-onboard
    ```
  </Tab>
  <Tab title="install.sh (non-interactive git)">
    ```bash
    MARV_INSTALL_METHOD=git MARV_NO_ONBOARD=1 \
      curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash
    ```
  </Tab>
  <Tab title="install-cli.sh (JSON)">
    ```bash
    curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install-cli.sh | bash -s -- --json --prefix /opt/marv
    ```
  </Tab>
  <Tab title="install.ps1 (skip onboarding)">
    ```powershell
    & ([scriptblock]::Create((iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1))) -NoOnboard
    ```
  </Tab>
</Tabs>

---

## Troubleshooting

<AccordionGroup>
  <Accordion title="Why is Git required?">
    Git is required for `git` install method. For `npm` installs, Git is still checked/installed to avoid `spawn git ENOENT` failures when dependencies use git URLs.
  </Accordion>

  <Accordion title="Why does npm hit EACCES on Linux?">
    Some Linux setups point npm global prefix to root-owned paths. `install.sh` can switch prefix to `~/.npm-global` and append PATH exports to shell rc files (when those files exist).
  </Accordion>

  <Accordion title="sharp/libvips issues">
    The scripts default `SHARP_IGNORE_GLOBAL_LIBVIPS=1` to avoid sharp building against system libvips. To override:

    ```bash
    SHARP_IGNORE_GLOBAL_LIBVIPS=0 curl -fsSL --proto '=https' --tlsv1.2 https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.sh | bash
    ```

  </Accordion>

  <Accordion title='Windows: "npm error spawn git / ENOENT"'>
    Install Git for Windows, reopen PowerShell, rerun installer.
  </Accordion>

  <Accordion title='Windows: "marv is not recognized"'>
    Run `npm config get prefix`, append `\bin`, add that directory to user PATH, then reopen PowerShell.
  </Accordion>

  <Accordion title="Windows: how to get verbose installer output">
    `install.ps1` does not currently expose a `-Verbose` switch.
    Use PowerShell tracing for script-level diagnostics:

    ```powershell
    Set-PSDebug -Trace 1
    & ([scriptblock]::Create((iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1))) -NoOnboard
    Set-PSDebug -Trace 0
    ```

  </Accordion>

  <Accordion title="marv not found after install">
    Usually a PATH issue. See [Node.js troubleshooting](/install/node#troubleshooting).
  </Accordion>
</AccordionGroup>

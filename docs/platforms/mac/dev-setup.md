---
summary: "Setup guide for developers working on the Marv macOS app"
read_when:
  - Setting up the macOS development environment
title: "macOS Dev Setup"
---

# macOS Developer Setup

This guide covers the necessary steps to build and run the Marv macOS application from source.

## Prerequisites

Before building the app, ensure you have the following installed:

1. **Xcode Command Line Tools** (minimum) or **Xcode 26.2+** (full IDE): Required for Swift compilation. Install with `xcode-select --install`.
2. **Node.js 22+ & pnpm**: Required for the gateway, CLI, and packaging scripts.

<Note>
No Apple Developer account or paid membership is required. The build script uses **ad-hoc signing** automatically when no Developer ID certificate is found.
</Note>

## Quick start (one command)

If you have already built the CLI (`pnpm install && pnpm build`), run:

```bash
scripts/setup-mac-app.sh
```

This builds the Swift app from source, installs it to `/Applications/Marv.app`, removes the Gatekeeper quarantine flag, and launches it. The menu-bar icon should appear shortly after.

## Step-by-step build

### 1. Install Dependencies

Install the project-wide dependencies:

```bash
pnpm install
```

### 2. Build and Package the App

To build the macOS app and package it into `dist/Marv.app`, run:

```bash
./scripts/package-mac-app.sh
```

If you don't have an Apple Developer ID certificate, the script will automatically use **ad-hoc signing** (`-`).

For dev run modes, signing flags, and Team ID troubleshooting, see the macOS app README:
[apps/macos/README.md](apps/macos/README.md)

> **Note**: Ad-hoc signed apps may trigger security prompts. If the app crashes immediately with "Abort trap 6", see the [Troubleshooting](#troubleshooting) section.

### 3. Install the CLI

The macOS app expects a global `marv` CLI install to manage background tasks.

**To install it (recommended):**

1. Open the Marv app.
2. Go to the **General** settings tab.
3. Click **"Install CLI"**.

Alternatively, install it manually:

```bash
npm install -g agentmarv@<version>
```

## Troubleshooting

### Build Fails: Toolchain or SDK Mismatch

The macOS app build expects the latest macOS SDK and Swift 6.2 toolchain.

**System dependencies (required):**

- **Latest macOS version available in Software Update** (required by Xcode 26.2 SDKs)
- **Xcode 26.2** (Swift 6.2 toolchain)

**Checks:**

```bash
xcodebuild -version
xcrun swift --version
```

If versions don’t match, update macOS/Xcode and re-run the build.

### App Crashes on Permission Grant

If the app crashes when you try to allow **Speech Recognition** or **Microphone** access, it may be due to a corrupted TCC cache or signature mismatch.

**Fix:**

1. Reset the TCC permissions:

   ```bash
   tccutil reset All ai.marv.mac.debug
   ```

2. If that fails, change the `BUNDLE_ID` temporarily in [`scripts/package-mac-app.sh`](scripts/package-mac-app.sh) to force a "clean slate" from macOS.

### Gateway "Starting..." indefinitely

If the gateway status stays on "Starting...", check if a zombie process is holding the port:

```bash
marv gateway status
marv gateway stop

# If you’re not using a LaunchAgent (dev mode / manual runs), find the listener:
lsof -nP -iTCP:4242 -sTCP:LISTEN
```

If a manual run is holding the port, stop that process (Ctrl+C). As a last resort, kill the PID you found above.

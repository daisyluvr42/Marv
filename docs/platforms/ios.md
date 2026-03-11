---
summary: "iOS companion app: chat, dashboard, voice input, and optional camera snapshots"
read_when:
  - Pairing or reconnecting the iOS companion app
  - Running the iOS app from source
  - Enabling optional camera snapshots for the agent
title: "iOS App"
---

# iOS App

Availability: internal preview. The iOS app is not publicly distributed yet.

## What it is

- Connects to a Gateway over WebSocket as an operator companion.
- Shows native chat and dashboard state from the Gateway.
- Supports on-device speech-to-text input that is sent into a normal chat session.
- Can optionally open a separate node connection for `camera.list` + `camera.snap`.

## Boundaries

- App Store distribution
- Background sensing or always-on listening
- Canvas, screen capture, or broad device control
- Continuous camera access

## Before you start

- Full Xcode installed
- `xcodegen` available on your PATH
- A real iPhone connected to this Mac
- A reachable `ws://` or `wss://` Gateway URL
- A shared Gateway token or password for the first connection

If `xcodebuild` still resolves to Command Line Tools, switch the active developer directory first:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

## Local deploy from source

Fast path:

```bash
pnpm ios:open
```

This runs local signing setup, generates the Xcode project, and opens it.
Choose a real iPhone in Xcode and run the `MarvCompanion` scheme.

If you only want to generate the project without opening Xcode:

```bash
pnpm ios:generate
```

Optional signing overrides:

```bash
MARV_IOS_APP_BUNDLE_ID=ai.marv.ios.myphone pnpm ios:open
```

The helper scripts auto-detect your Apple Development team ID and write a local signing file under `apps/ios/.local-signing.xcconfig`.
Advanced overrides:

- `MARV_IOS_APP_BUNDLE_ID`
- `MARV_IOS_CODE_SIGN_STYLE`
- `MARV_IOS_APP_PROFILE`

## Pair as an operator companion

1. Start the Gateway.
2. In the app Settings tab, enter the Gateway URL and shared token/password.
3. Tap **Save And Connect**.
4. Approve the first device pairing request for the `operator` role.
5. Reconnect after approval to activate chat and dashboard.

Example approval flow:

```bash
marv devices list
marv devices approve <requestId>
```

## Use it as a display + input terminal

- **Chat** shows a normal operator session backed by the Gateway.
- **Dashboard** shows the same Memory / Knowledge / Proactive status cards as the browser Control UI.
- **Voice input** uses on-device speech recognition and sends the transcript into the current session.

This makes the app a good fit for an idle iPhone that mainly sits on a desk as a companion screen and input endpoint.

## Optional camera snapshots for the agent

The camera path is deliberately separate from the operator path.

1. In the app Settings tab, enable **Enable agent camera snapshots**.
2. Reconnect the app.
3. Approve the new pairing request for the `node` role if prompted.
4. Allowlist the dangerous camera command on the Gateway:

```bash
marv config set gateway.nodes.allowCommands '["camera.snap"]'
```

The current repo app declares `camera.list` and `camera.snap` only. It does not keep the camera open, and snapshot requests only succeed while the app is in the foreground.

## Common errors

- `xcodegen is required`: install it with `brew install xcodegen`, then run `pnpm ios:open` again.
- `xcodebuild is pointing at CommandLineTools`: switch to full Xcode with `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.
- `pairing required`: approve the pending device request, then reconnect.
- `missing scope: operator.read` or `operator.write`: reconnect after the operator-role pairing approval is complete.
- `command not allowlisted`: add `camera.snap` to `gateway.nodes.allowCommands`.
- `NODE_BACKGROUND_UNAVAILABLE`: bring the app to the foreground before asking the agent to take a snapshot.

## Related docs

- [Dashboard](/web/dashboard)
- [Pairing](/gateway/pairing)
- [Camera Capture](/nodes/camera)
- [Remote Access](/gateway/remote)

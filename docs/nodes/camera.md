---
summary: "Camera capture (iOS node + macOS app) for agent use: photos (jpg) and short video clips (mp4)"
read_when:
  - Adding or modifying camera capture on iOS nodes or macOS
  - Extending agent-accessible MEDIA temp-file workflows
title: "Camera Capture"
---

# Camera capture (agent)

Marv supports **camera capture** for agent workflows:

- **iOS companion app** (paired via Gateway): capture a **photo** (`jpg`) via `node.invoke`.
- **Android node** (paired via Gateway): capture a **photo** (`jpg`) or **short video clip** (`mp4`, with optional audio) via `node.invoke`.
- **macOS app** (node via Gateway): capture a **photo** (`jpg`) or **short video clip** (`mp4`, with optional audio) via `node.invoke`.

All camera access is gated behind **user-controlled settings**.

## iOS companion app

### User setting (default off)

- iOS Settings tab → **Enable agent camera snapshots**
  - Default: **off**
  - When off: the app does not open the node camera connection at all.
  - Turning it on usually creates a second pairing request because the device now asks for the `node` role.

### Commands (via Gateway `node.invoke`)

- `camera.list`
  - Response payload:
    - `devices`: array of `{ id, name, position, deviceType }`

- `camera.snap`
  - Params:
    - `facing`: `front|back` (default: `front`)
    - `maxWidth`: number (optional; default `1600` on the iOS node)
    - `quality`: `0..1` (optional; default `0.9`)
    - `format`: currently `jpg`
    - `delayMs`: number (optional; default `0`)
    - `deviceId`: string (optional; from `camera.list`)
  - Response payload:
    - `format: "jpg"`
    - `base64: "<...>"`
    - `width`, `height`
  - Payload guard: photos are recompressed to keep the base64 payload under 5 MB.

Current repo status:

- `camera.clip` is **not** wired in the iOS companion app.
- The app only declares `camera.list` and `camera.snap`.
- The Gateway still treats `camera.snap` as a dangerous node command, so you must explicitly allowlist it with `gateway.nodes.allowCommands`.

### Foreground requirement

The iOS companion app only allows camera snapshots while it is in the **foreground**. Background invocations return `NODE_BACKGROUND_UNAVAILABLE`.

### CLI helper (temp files + MEDIA)

The easiest way to get attachments is via the CLI helper, which writes decoded media to a temp file and prints `MEDIA:<path>`.

Examples:

```bash
marv nodes camera snap --node <id>               # default: both front + back (2 MEDIA lines)
marv nodes camera snap --node <id> --facing front
```

Notes:

- The iOS companion app exposes only still-photo capture right now.
- `nodes camera snap` defaults to **both** facings to give the agent both views.
- Output files are temporary (in the OS temp directory) unless you build your own wrapper.

## Android node

### Android user setting (default on)

- Android Settings sheet → **Camera** → **Allow Camera** (`camera.enabled`)
  - Default: **on** (missing key is treated as enabled).
  - When off: `camera.*` commands return `CAMERA_DISABLED`.

### Permissions

- Android requires runtime permissions:
  - `CAMERA` for both `camera.snap` and `camera.clip`.
  - `RECORD_AUDIO` for `camera.clip` when `includeAudio=true`.

If permissions are missing, the app will prompt when possible; if denied, `camera.*` requests fail with a
`*_PERMISSION_REQUIRED` error.

### Android foreground requirement

Like `canvas.*`, the Android node only allows `camera.*` commands in the **foreground**. Background invocations return `NODE_BACKGROUND_UNAVAILABLE`.

### Payload guard

Photos are recompressed to keep the base64 payload under 5 MB.

## macOS app

### User setting (default off)

The macOS companion app exposes a checkbox:

- **Settings → General → Allow Camera** (`marv.cameraEnabled`)
  - Default: **off**
  - When off: camera requests return “Camera disabled by user”.

### CLI helper (node invoke)

Use the main `marv` CLI to invoke camera commands on the macOS node.

Examples:

```bash
marv nodes camera list --node <id>            # list camera ids
marv nodes camera snap --node <id>            # prints MEDIA:<path>
marv nodes camera snap --node <id> --max-width 1280
marv nodes camera snap --node <id> --delay-ms 2000
marv nodes camera snap --node <id> --device-id <id>
marv nodes camera clip --node <id> --duration 10s          # prints MEDIA:<path>
marv nodes camera clip --node <id> --duration-ms 3000      # prints MEDIA:<path> (legacy flag)
marv nodes camera clip --node <id> --device-id <id>
marv nodes camera clip --node <id> --no-audio
```

Notes:

- `marv nodes camera snap` defaults to `maxWidth=1600` unless overridden.
- On macOS, `camera.snap` waits `delayMs` (default 2000ms) after warm-up/exposure settle before capturing.
- Photo payloads are recompressed to keep base64 under 5 MB.

## Safety + practical limits

- Camera and microphone access trigger the usual OS permission prompts (and require usage strings in Info.plist).
- Video clips are capped (currently `<= 60s`) to avoid oversized node payloads (base64 overhead + message limits).

## macOS screen video (OS-level)

For _screen_ video (not camera), use the macOS companion:

```bash
marv nodes screen record --node <id> --duration 10s --fps 15   # prints MEDIA:<path>
```

Notes:

- Requires macOS **Screen Recording** permission (TCC).

# Marv iOS Companion

This app is a local/manual deploy companion for Marv.

Current scope:

- Gateway chat display
- Gateway dashboard status cards
- On-device speech-to-text input
- Optional camera node access for `camera.snap`

Non-goals for this build:

- App Store distribution
- Background sensing
- Canvas, screen capture, or broad device control

Quick start:

```bash
pnpm ios:open
```

Prerequisites:

- Full Xcode installed
- `xcodegen` installed (`brew install xcodegen`)

If `xcodebuild` still points at Command Line Tools, switch it first:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

Then pick a real iPhone in Xcode and run the `MarvCompanion` scheme.

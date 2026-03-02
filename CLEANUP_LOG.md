# Marv Phase 1 Cleanup Log

Date: 2026-03-02

## Scope

Phase 1 (Route B) aggressive cleanup for Marv fork baseline.
Principle: prefer deletion over retention for unused upstream surfaces.

## Decisions

### 1) Native apps and Apple release artifacts

- Delete `apps/` and Apple tooling files (`.swiftformat`, `.swiftlint.yml`, `appcast.xml`).
- Reason: Marv is being narrowed to gateway + CLI; native clients are out of scope for Phase 1.

### 2) Swabble

- Delete `Swabble/`.
- Reason: upstream internal/experimental project, not needed for Marv runtime.

### 3) Sandbox Docker images

- Delete `Dockerfile.sandbox`, `Dockerfile.sandbox-browser`, `Dockerfile.sandbox-common`.
- Reason: single-user setup does not require dedicated Docker sandbox image stack in this phase.

### 4) Deployment platform leftovers

- Evaluate and remove Fly/Render/Podman setup files if unused by Marv current workflow.

### 5) Upstream entry/branding leftovers

- Delete `openclaw.mjs` if present.
- Replace remaining upstream OpenClaw branding references in first-party code/docs where safe, excluding LICENSE/CHANGELOG/third-party package names/history.

### 6) Test config simplification

- Consolidate Vitest configs to max two files: default + e2e.

### 7) Root misc + vendor + CI/scripts hygiene

- Remove root-level and automation artifacts that no longer match active Marv workflow.
- Keep core runtime and Marv-specific docs/assets.

### 9) GitHub automation trim

- Remove CI/workflows tied to native apps and sandbox image release lanes.
- Keep only core Node checks and secret scan.

### 10) Agent/internal workflow trim

- Remove archived `.agents/archive` legacy workflow snapshots.
- Keep active `.agents/skills/*` and `.pi/*` because they are still used by local coding-agent tooling.

### 11) Scripts + package commands trim

- Remove scripts for macOS app packaging/signing/notarization, iOS tooling, Swift protocol generation, and sandbox image setup.
- Remove corresponding npm scripts from `package.json`; keep CLI/gateway/build/test scripts.

### 8) Validation policy

- Run `pnpm build` after each major deletion category.
- Run `pnpm test` at end; remove tests that target removed functionality if needed.

## Execution Log

### Completed: Native app surface removal

- Deleted entire `apps/` tree.
- Deleted `.swiftformat`, `.swiftlint.yml`, and `appcast.xml`.
- Build validation: `pnpm build` passed.

### Build-health fixes applied during validation

- Removed duplicate declarations/properties that caused parser/typecheck failures in:
  - `src/version.ts`
  - `src/infra/tmp-marv-dir.ts`
  - `src/plugins/manifest.ts`
  - `src/agents/marv-tools.ts`
  - `src/browser/constants.ts`
  - `src/browser/config.ts`
  - `src/infra/path-env.ts`
  - `src/hooks/types.ts`
  - `src/agents/sandbox/browser.ts`
  - `src/agents/tool-policy.ts`
  - `src/browser/extension-relay.ts`
  - `src/gateway/session-utils.fs.ts`
  - `src/plugins/loader.ts`
- Reason: these duplicates blocked required build checks and were minimal no-behavior cleanup edits.

### Completed: Swabble removal

- Deleted `Swabble/`.
- Build validation: `pnpm build` passed.

### Completed: Sandbox Dockerfiles removal

- Deleted `Dockerfile.sandbox`, `Dockerfile.sandbox-browser`, `Dockerfile.sandbox-common`.
- Build validation: `pnpm build` passed.

### Completed: Deploy config cleanup

- Deleted `fly.toml`, `fly.private.toml`, `render.yaml`, `setup-podman.sh`, `marv.podman.env`.
- Build validation: `pnpm build` passed.

### Completed: Vitest config consolidation

- Removed split configs (`vitest.unit.config.ts`, `vitest.gateway.config.ts`, `vitest.extensions.config.ts`, `vitest.live.config.ts`).
- Kept only `vitest.config.ts` (projects) and `vitest.e2e.config.ts`.
- Updated scripts to run via `--project`.

### Completed: Root misc cleanup

- Removed `.pre-commit-config.yaml`, `docs.acp.md`, `zizmor.yml`.
- Kept detect-secrets, markdownlint, shellcheck, oxfmt/oxlint configs, and `tsconfig.plugin-sdk.dts.json` because they remain referenced.

### Completed: GitHub cleanup

- Removed app/sandbox-heavy workflows and simplified CI to lint/build/test/secrets.
- Trimmed dependabot entries for Swift/Gradle and removed app labels from labeler config.

### Completed: Scripts and command cleanup

- Removed archived `.agents/archive`.
- Removed native app + sandbox setup scripts from `scripts/`.
- Updated `package.json` scripts to drop `ios:*`, `android:*`, `mac:*`, `*:swift`, and `protocol:gen:swift`.
- Build validation: `pnpm build` passed.

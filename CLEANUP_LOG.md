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

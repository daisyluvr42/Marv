#!/usr/bin/env -S node --import tsx

/**
 * Interactive release pipeline for Marv.
 *
 * Orchestrates: version bump → build → validate → mac app → commit/tag →
 * npm publish → GitHub release → post-publish verification.
 *
 * Usage:
 *   pnpm release              # full interactive release
 *   pnpm release --skip-mac   # skip macOS app packaging
 *   pnpm release --dry-run    # run all steps except actual publish
 */

import { execSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = resolve(import.meta.dirname, "..");

// ── CLI args ──────────────────────────────────────────────────────────

const args = new Set(process.argv.slice(2));
const skipMac = args.has("--skip-mac");
const dryRun = args.has("--dry-run");

// ── State tracking for checkpoint/resume ──────────────────────────────

type ReleaseState = {
  version?: string;
  completedSteps: string[];
};

const STATE_FILE = resolve(ROOT, ".release-state.json");

function loadState(): ReleaseState {
  try {
    if (existsSync(STATE_FILE)) {
      return JSON.parse(readFileSync(STATE_FILE, "utf8")) as ReleaseState;
    }
  } catch {
    // ignore
  }
  return { completedSteps: [] };
}

function saveState(state: ReleaseState): void {
  writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

function clearState(): void {
  try {
    if (existsSync(STATE_FILE)) {
      const { unlinkSync } = require("node:fs");
      unlinkSync(STATE_FILE);
    }
  } catch {
    // ignore
  }
}

// ── Helpers ───────────────────────────────────────────────────────────

function run(cmd: string, opts?: { cwd?: string; stdio?: "inherit" | "pipe" }): string {
  const stdio = opts?.stdio ?? "inherit";
  if (stdio === "pipe") {
    return execSync(cmd, { cwd: opts?.cwd ?? ROOT, encoding: "utf8", stdio: "pipe" }).trim();
  }
  execSync(cmd, { cwd: opts?.cwd ?? ROOT, stdio: "inherit" });
  return "";
}

function confirm(message: string): boolean {
  if (dryRun) {
    console.log(`  [dry-run] Would ask: ${message}`);
    return true;
  }
  const result = spawnSync(
    "bash",
    ["-c", `read -rp "${message} [y/N] " ans && [[ "$ans" =~ ^[Yy] ]]`],
    {
      stdio: "inherit",
    },
  );
  return result.status === 0;
}

function readVersion(): string {
  const pkg = JSON.parse(readFileSync(resolve(ROOT, "package.json"), "utf8")) as {
    version: string;
  };
  return pkg.version;
}

function heading(text: string): void {
  console.log(`\n${"─".repeat(60)}\n  ${text}\n${"─".repeat(60)}`);
}

// ── Steps ─────────────────────────────────────────────────────────────

type Step = {
  id: string;
  label: string;
  run: (state: ReleaseState) => void;
};

const steps: Step[] = [
  {
    id: "preflight",
    label: "Preflight checks",
    run() {
      heading("Preflight");

      // Check git clean
      const status = run("git status --porcelain", { stdio: "pipe" });
      if (status) {
        console.error("Working tree is dirty. Commit or stash changes first.");
        process.exit(1);
      }

      // Check env vars
      const sparkleKey = process.env.SPARKLE_PRIVATE_KEY_FILE;
      if (!skipMac && !sparkleKey) {
        console.warn("Warning: SPARKLE_PRIVATE_KEY_FILE not set. macOS app packaging may fail.");
      }

      console.log("  Git: clean");
      console.log(`  SPARKLE_PRIVATE_KEY_FILE: ${sparkleKey ? "set" : "NOT SET"}`);
      console.log(`  Current version: ${readVersion()}`);
    },
  },
  {
    id: "version-bump",
    label: "Version bump",
    run(state) {
      heading("Version Bump");

      if (state.version) {
        console.log(`  Using previously set version: ${state.version}`);
        run(`pnpm version:bump ${state.version}`);
      } else {
        console.log("  Run: pnpm version:bump <version>");
        console.log("  Enter the target version when prompted.");
        // In a real interactive flow, we'd prompt here.
        // For now, require the version in state or via env.
        const version = process.env.RELEASE_VERSION;
        if (!version) {
          console.error("Set RELEASE_VERSION env var or resume with a saved state.");
          process.exit(1);
        }
        run(`pnpm version:bump ${version}`);
        state.version = version;
        saveState(state);
      }
    },
  },
  {
    id: "build",
    label: "Build & validate",
    run() {
      heading("Build & Validate");
      run("pnpm install");
      run("pnpm build");
      run("pnpm check");
      console.log("\n  Build and lint passed.");
    },
  },
  {
    id: "test",
    label: "Run tests",
    run() {
      heading("Tests");
      run("pnpm test");
      console.log("\n  Tests passed.");
    },
  },
  {
    id: "release-check",
    label: "Release check (npm pack validation)",
    run() {
      heading("Release Check");
      run("pnpm release:check");
      console.log("  npm pack contents verified.");
    },
  },
  {
    id: "changelog",
    label: "Changelog review",
    run(state) {
      heading("Changelog");
      const version = state.version ?? readVersion();
      const changelogPath = resolve(ROOT, "CHANGELOG.md");
      if (existsSync(changelogPath)) {
        const content = readFileSync(changelogPath, "utf8");
        if (!content.includes(`## ${version}`)) {
          console.warn(`  Warning: CHANGELOG.md does not contain a section for ${version}`);
          console.warn("  Please add changelog entries before publishing.");
        } else {
          console.log(`  Changelog entry found for ${version}.`);
        }
      } else {
        console.warn("  CHANGELOG.md not found.");
      }
    },
  },
  {
    id: "mac-app",
    label: "macOS app packaging",
    run(state) {
      if (skipMac) {
        console.log("  Skipped (--skip-mac).");
        return;
      }
      heading("macOS App");
      const version = state.version ?? readVersion();
      console.log(`  Building macOS app for version ${version}...`);
      console.log("  Run: scripts/package-mac-dist.sh with appropriate env vars.");
      console.log("  Run: scripts/make_appcast.sh to update appcast.xml.");
      console.log("  (This step requires manual execution — see docs/platforms/mac/release.md)");
    },
  },
  {
    id: "commit-tag",
    label: "Commit & tag",
    run(state) {
      heading("Commit & Tag");
      const version = state.version ?? readVersion();
      const tag = `v${version}`;

      if (dryRun) {
        console.log(`  [dry-run] Would commit and tag ${tag}`);
        return;
      }

      if (!confirm(`Commit and tag ${tag}?`)) {
        console.log("  Skipped.");
        return;
      }

      run("git add -A");
      run(`git commit -m "Release ${version}"`);
      run(`git tag ${tag}`);
      console.log(`  Created tag ${tag}.`);
    },
  },
  {
    id: "npm-publish",
    label: "Publish to npm",
    run(state) {
      heading("npm Publish");
      const version = state.version ?? readVersion();

      if (dryRun) {
        console.log(`  [dry-run] Would publish agentmarv@${version}`);
        return;
      }

      if (!confirm(`Publish agentmarv@${version} to npm?`)) {
        console.log("  Skipped.");
        return;
      }

      const isBeta = version.includes("-beta");
      const tagFlag = isBeta ? " --tag beta" : "";
      console.log("  Publishing... (you may need to provide an OTP)");
      run(`npm publish --access public${tagFlag}`);
      console.log(`  Published agentmarv@${version}.`);
    },
  },
  {
    id: "github-release",
    label: "Create GitHub release",
    run(state) {
      heading("GitHub Release");
      const version = state.version ?? readVersion();
      const tag = `v${version}`;

      if (dryRun) {
        console.log(`  [dry-run] Would push tags and create GitHub release for ${tag}`);
        return;
      }

      if (!confirm(`Push tag ${tag} and create GitHub release?`)) {
        console.log("  Skipped.");
        return;
      }

      run(`git push origin ${tag}`);
      run("git push");

      const isBeta = version.includes("-beta");
      const prerelease = isBeta ? " --prerelease" : "";
      run(`gh release create ${tag} --title "marv ${version}"${prerelease} --generate-notes`);
      console.log(`  GitHub release created for ${tag}.`);
    },
  },
  {
    id: "verify",
    label: "Post-publish verification",
    run() {
      heading("Verification");

      if (dryRun) {
        console.log("  [dry-run] Would verify npm publication.");
        return;
      }

      try {
        const npmVersion = run('npm view agentmarv version --userconfig "$(mktemp)"', {
          stdio: "pipe",
        });
        console.log(`  npm registry version: ${npmVersion}`);
      } catch {
        console.warn("  Warning: Could not verify npm version.");
      }

      console.log("  Release complete!");
    },
  },
];

// ── Main ──────────────────────────────────────────────────────────────

function main() {
  console.log("Marv Release Pipeline");
  console.log(`Mode: ${dryRun ? "DRY RUN" : "LIVE"}`);
  if (skipMac) {
    console.log("macOS app packaging: SKIPPED");
  }

  const state = loadState();

  for (const step of steps) {
    if (state.completedSteps.includes(step.id)) {
      console.log(`  [skip] ${step.label} (already completed)`);
      continue;
    }

    try {
      step.run(state);
      state.completedSteps.push(step.id);
      saveState(state);
    } catch (err) {
      console.error(`\nStep "${step.label}" failed.`);
      console.error(err instanceof Error ? err.message : String(err));
      console.error("\nFix the issue, then re-run `pnpm release` to resume from this step.");
      process.exit(1);
    }
  }

  clearState();
  console.log("\nRelease pipeline complete.");
}

main();

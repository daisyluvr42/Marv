#!/usr/bin/env -S node --import tsx

import { execSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

// Version format: YYYY.M.D or YYYY.M.D-beta.N
const VERSION_RE = /^\d{4}\.\d{1,2}\.\d{1,2}(-beta\.\d+)?$/;

type VersionTarget = {
  file: string;
  label: string;
  update: (content: string, version: string) => string;
};

// ── plist helpers ──────────────────────────────────────────────────────

function plistReplace(content: string, key: string, value: string): string {
  // Match <key>KEY</key> followed by <string>...</string> on the next line
  const re = new RegExp(`(<key>${escapeRegExp(key)}</key>\\s*\\n\\s*<string>)([^<]*)(</string>)`);
  if (!re.test(content)) {
    throw new Error(`plist key "${key}" not found`);
  }
  return content.replace(re, `$1${value}$3`);
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ── version format converters ─────────────────────────────────────────

/** "2026.3.17" → "20260317" (iOS CFBundleVersion format) */
function toBundleVersionIos(version: string): string {
  const base = version.replace(/-beta\.\d+$/, "");
  const [y, m, d] = base.split(".");
  return `${y}${m.padStart(2, "0")}${d.padStart(2, "0")}`;
}

/** "2026.3.17" → "202603170" (macOS/Android CFBundleVersion format, trailing 0) */
function toBundleVersionMac(version: string): string {
  return `${toBundleVersionIos(version)}0`;
}

// ── targets ───────────────────────────────────────────────────────────

const ROOT = resolve(import.meta.dirname, "..");

function buildTargets(): VersionTarget[] {
  return [
    {
      file: resolve(ROOT, "package.json"),
      label: "package.json",
      update(content, version) {
        const pkg = JSON.parse(content) as { version?: string };
        pkg.version = version;
        return `${JSON.stringify(pkg, null, 2)}\n`;
      },
    },
    {
      file: resolve(ROOT, "apps/macos/Sources/Marv/Resources/Info.plist"),
      label: "macOS Info.plist",
      update(content, version) {
        let result = plistReplace(content, "CFBundleShortVersionString", version);
        result = plistReplace(result, "CFBundleVersion", toBundleVersionMac(version));
        return result;
      },
    },
    {
      file: resolve(ROOT, "apps/ios/Sources/Info.plist"),
      label: "iOS Info.plist",
      update(content, version) {
        let result = plistReplace(content, "CFBundleShortVersionString", version);
        result = plistReplace(result, "CFBundleVersion", toBundleVersionIos(version));
        return result;
      },
    },
  ];
}

// ── main ──────────────────────────────────────────────────────────────

function parseArgs(): { version: string | null; dryRun: boolean } {
  const args = process.argv.slice(2);
  let dryRun = false;
  let version: string | null = null;

  for (const arg of args) {
    if (arg === "--dry-run") {
      dryRun = true;
    } else if (arg === "--help" || arg === "-h") {
      console.log("Usage: bump-version [VERSION] [--dry-run]");
      console.log("  VERSION   e.g. 2026.3.17 or 2026.3.17-beta.1");
      console.log("  --dry-run Show changes without writing files");
      process.exit(0);
    } else if (!version && !arg.startsWith("-")) {
      version = arg;
    }
  }

  return { version, dryRun };
}

function readCurrentVersion(): string {
  const pkg = JSON.parse(readFileSync(resolve(ROOT, "package.json"), "utf8")) as {
    version?: string;
  };
  return pkg.version ?? "0.0.0";
}

function main() {
  const { version: requestedVersion, dryRun } = parseArgs();

  const currentVersion = readCurrentVersion();

  if (!requestedVersion) {
    console.error(`Current version: ${currentVersion}`);
    console.error("Usage: bump-version <VERSION> [--dry-run]");
    process.exit(1);
  }

  if (!VERSION_RE.test(requestedVersion)) {
    console.error(`Invalid version format: "${requestedVersion}"`);
    console.error("Expected: YYYY.M.D or YYYY.M.D-beta.N (e.g. 2026.3.17)");
    process.exit(1);
  }

  const targets = buildTargets();
  const results: { label: string; before: string; after: string; written: boolean }[] = [];

  for (const target of targets) {
    let content: string;
    try {
      content = readFileSync(target.file, "utf8");
    } catch {
      results.push({
        label: target.label,
        before: "(missing)",
        after: "(skipped)",
        written: false,
      });
      continue;
    }

    const updated = target.update(content, requestedVersion);
    const changed = content !== updated;

    if (changed && !dryRun) {
      writeFileSync(target.file, updated);
    }

    results.push({
      label: target.label,
      before: currentVersion,
      after: changed ? requestedVersion : "(unchanged)",
      written: changed && !dryRun,
    });
  }

  // Sync plugin versions (unless dry-run)
  let pluginResult = "(skipped)";
  if (!dryRun) {
    try {
      const syncScript = resolve(ROOT, "scripts/sync-plugin-versions.ts");
      execSync(`node --import tsx "${syncScript}"`, {
        cwd: ROOT,
        stdio: "pipe",
        encoding: "utf8",
      });
      pluginResult = "synced";
    } catch (err) {
      pluginResult = `error: ${err instanceof Error ? err.message : String(err)}`;
    }
  } else {
    pluginResult = "(dry-run)";
  }

  // Print summary
  console.log("");
  console.log(dryRun ? "Dry run — no files written:" : "Version bump complete:");
  console.log("");
  const pad = Math.max(...results.map((r) => r.label.length));
  for (const r of results) {
    const status = r.written
      ? "updated"
      : dryRun && r.after !== "(skipped)" && r.after !== "(unchanged)"
        ? "would update"
        : r.after;
    console.log(`  ${r.label.padEnd(pad)}  ${r.before} -> ${r.after}  [${status}]`);
  }
  console.log(`  ${"extensions/*".padEnd(pad)}  ${pluginResult}`);
  console.log("");

  if (!dryRun) {
    console.log(`Bumped all version locations to ${requestedVersion}.`);
    console.log("Next: review changes, then commit.");
  }
}

main();

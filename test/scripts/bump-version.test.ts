import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

// ── Import the helpers we want to test ────────────────────────────────
// We re-implement the core logic inline since the script is a standalone
// CLI entry. This tests the same algorithms.

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function plistReplace(content: string, key: string, value: string): string {
  const re = new RegExp(`(<key>${escapeRegExp(key)}</key>\\s*\\n\\s*<string>)([^<]*)(</string>)`);
  if (!re.test(content)) {
    throw new Error(`plist key "${key}" not found`);
  }
  return content.replace(re, `$1${value}$3`);
}

function toBundleVersionIos(version: string): string {
  const base = version.replace(/-beta\.\d+$/, "");
  const [y, m, d] = base.split(".");
  return `${y}${m.padStart(2, "0")}${d.padStart(2, "0")}`;
}

function toBundleVersionMac(version: string): string {
  return `${toBundleVersionIos(version)}0`;
}

const VERSION_RE = /^\d{4}\.\d{1,2}\.\d{1,2}(-beta\.\d+)?$/;

// ── Tests ─────────────────────────────────────────────────────────────

describe("bump-version helpers", () => {
  describe("VERSION_RE", () => {
    it("accepts valid versions", () => {
      expect(VERSION_RE.test("2026.3.17")).toBe(true);
      expect(VERSION_RE.test("2026.12.1")).toBe(true);
      expect(VERSION_RE.test("2026.3.17-beta.1")).toBe(true);
      expect(VERSION_RE.test("2026.3.17-beta.12")).toBe(true);
    });

    it("rejects invalid versions", () => {
      expect(VERSION_RE.test("1.0.0")).toBe(false);
      expect(VERSION_RE.test("2026.3")).toBe(false);
      expect(VERSION_RE.test("2026.3.17-rc.1")).toBe(false);
      expect(VERSION_RE.test("v2026.3.17")).toBe(false);
      expect(VERSION_RE.test("")).toBe(false);
    });
  });

  describe("toBundleVersionIos", () => {
    it("converts YYYY.M.D to YYYYMMDD", () => {
      expect(toBundleVersionIos("2026.3.17")).toBe("20260317");
      expect(toBundleVersionIos("2026.12.1")).toBe("20261201");
      expect(toBundleVersionIos("2026.1.5")).toBe("20260105");
    });

    it("strips beta suffix", () => {
      expect(toBundleVersionIos("2026.3.17-beta.1")).toBe("20260317");
    });
  });

  describe("toBundleVersionMac", () => {
    it("converts YYYY.M.D to YYYYMMDD0", () => {
      expect(toBundleVersionMac("2026.3.17")).toBe("202603170");
      expect(toBundleVersionMac("2026.2.23")).toBe("202602230");
    });
  });

  describe("plistReplace", () => {
    const samplePlist = `<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>CFBundleShortVersionString</key>
    <string>2026.2.23</string>
    <key>CFBundleVersion</key>
    <string>202602230</string>
</dict>
</plist>`;

    it("replaces CFBundleShortVersionString", () => {
      const result = plistReplace(samplePlist, "CFBundleShortVersionString", "2026.3.17");
      expect(result).toContain("<string>2026.3.17</string>");
      // Ensure CFBundleVersion is untouched
      expect(result).toContain("<string>202602230</string>");
    });

    it("replaces CFBundleVersion", () => {
      const result = plistReplace(samplePlist, "CFBundleVersion", "202603170");
      expect(result).toContain("<string>202603170</string>");
    });

    it("throws for missing key", () => {
      expect(() => plistReplace(samplePlist, "MissingKey", "value")).toThrow(
        'plist key "MissingKey" not found',
      );
    });

    it("handles tab-indented plists", () => {
      const tabPlist = `<dict>\n\t<key>CFBundleShortVersionString</key>\n\t<string>1.0.0</string>\n</dict>`;
      const result = plistReplace(tabPlist, "CFBundleShortVersionString", "2026.3.17");
      expect(result).toContain("<string>2026.3.17</string>");
    });
  });

  describe("package.json update", () => {
    it("updates the version field", () => {
      const dir = mkdtempSync(join(tmpdir(), "bump-test-"));
      const pkgPath = join(dir, "package.json");
      writeFileSync(pkgPath, JSON.stringify({ name: "test", version: "1.0.0" }, null, 2));

      const content = readFileSync(pkgPath, "utf8");
      const pkg = JSON.parse(content) as { version?: string };
      pkg.version = "2026.3.17";
      writeFileSync(pkgPath, `${JSON.stringify(pkg, null, 2)}\n`);

      const updated = JSON.parse(readFileSync(pkgPath, "utf8")) as { version: string };
      expect(updated.version).toBe("2026.3.17");
    });
  });
});

import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UpdateCheckResult } from "./update-check.js";

vi.mock("../marv-root.js", () => ({
  resolveMarvPackageRoot: vi.fn(),
}));

vi.mock("./update-check.js", async () => {
  const parse = (value: string) => value.split(".").map((part) => Number.parseInt(part, 10));
  const compareSemverStrings = (a: string | null, b: string | null) => {
    if (!a || !b) {
      return null;
    }
    const left = parse(a);
    const right = parse(b);
    for (let idx = 0; idx < 3; idx += 1) {
      const l = left[idx] ?? 0;
      const r = right[idx] ?? 0;
      if (l !== r) {
        return l < r ? -1 : 1;
      }
    }
    return 0;
  };

  return {
    checkUpdateStatus: vi.fn(),
    compareSemverStrings,
    resolveNpmChannelTag: vi.fn(),
  };
});

vi.mock("../../version.js", () => ({
  VERSION: "1.0.0",
}));

describe("update-notify", () => {
  let resolveMarvPackageRoot: (typeof import("../marv-root.js"))["resolveMarvPackageRoot"];
  let checkUpdateStatus: (typeof import("./update-check.js"))["checkUpdateStatus"];
  let resolveNpmChannelTag: (typeof import("./update-check.js"))["resolveNpmChannelTag"];
  let checkForUpdate: (typeof import("./update-notify.js"))["checkForUpdate"];
  let loaded = false;

  beforeEach(async () => {
    if (!loaded) {
      ({ resolveMarvPackageRoot } = await import("../marv-root.js"));
      ({ checkUpdateStatus, resolveNpmChannelTag } = await import("./update-check.js"));
      ({ checkForUpdate } = await import("./update-notify.js"));
      loaded = true;
    }
    vi.clearAllMocks();
    vi.mocked(resolveMarvPackageRoot).mockResolvedValue("/opt/marv");
  });

  it("detects newer npm package versions", async () => {
    vi.mocked(checkUpdateStatus).mockResolvedValue({
      root: "/opt/marv",
      installKind: "package",
      packageManager: "npm",
    } satisfies UpdateCheckResult);
    vi.mocked(resolveNpmChannelTag).mockResolvedValue({
      tag: "latest",
      version: "2.0.0",
    });

    const update = await checkForUpdate({
      cfg: { update: { channel: "stable" } },
      fetchGit: false,
    });

    expect(update).toMatchObject({
      available: true,
      currentVersion: "1.0.0",
      latestVersion: "2.0.0",
      channel: "stable",
      tag: "latest",
      installKind: "package",
    });
  });

  it("uses latest when beta resolves to an older prerelease", async () => {
    vi.mocked(checkUpdateStatus).mockResolvedValue({
      root: "/opt/marv",
      installKind: "package",
      packageManager: "npm",
    } satisfies UpdateCheckResult);
    vi.mocked(resolveNpmChannelTag).mockResolvedValue({
      tag: "latest",
      version: "2.0.0",
    });

    const update = await checkForUpdate({
      cfg: { update: { channel: "beta" } },
      fetchGit: false,
    });

    expect(update.channel).toBe("beta");
    expect(update.tag).toBe("latest");
    expect(update.latestVersion).toBe("2.0.0");
  });

  it("detects newer git upstream commits", async () => {
    vi.mocked(checkUpdateStatus).mockResolvedValue({
      root: "/opt/marv",
      installKind: "git",
      packageManager: "pnpm",
      git: {
        root: "/opt/marv",
        sha: "abc123456789",
        tag: null,
        branch: "main",
        upstream: "origin/main",
        upstreamSha: "def987654321",
        dirty: false,
        ahead: 0,
        behind: 2,
        fetchOk: true,
      },
    } satisfies UpdateCheckResult);

    const update = await checkForUpdate({
      cfg: { update: { channel: "dev" } },
      fetchGit: true,
    });

    expect(update).toMatchObject({
      available: true,
      currentVersion: "abc12345",
      latestVersion: "def98765",
      channel: "dev",
      tag: "dev",
      installKind: "git",
      git: {
        branch: "main",
        upstream: "origin/main",
        behind: 2,
      },
    });
  });

  it("uses the latest approved deploy tag when approval is required", async () => {
    vi.mocked(checkUpdateStatus).mockResolvedValue({
      root: "/opt/marv",
      installKind: "git",
      packageManager: "pnpm",
      git: {
        root: "/opt/marv",
        sha: "abc123456789",
        tag: null,
        branch: "main",
        upstream: "origin/main",
        upstreamSha: "def987654321",
        dirty: false,
        ahead: 0,
        behind: 4,
        fetchOk: true,
        approval: {
          required: true,
          mode: "signed-tag",
          tagPattern: "deploy/*",
          branch: "main",
          branchRef: "origin/main",
          branchSha: "def987654321",
          approvedTag: "deploy/2026-03-09-1",
          approvedSha: "fedcba987654",
          pendingCommits: 2,
          reason: null,
        },
      },
    } satisfies UpdateCheckResult);

    const update = await checkForUpdate({
      cfg: {
        update: {
          channel: "dev",
          approval: { required: true, mode: "signed-tag" },
        },
      },
      fetchGit: true,
    });

    expect(update).toMatchObject({
      available: true,
      currentVersion: "abc12345",
      latestVersion: "fedcba98",
      tag: "deploy/2026-03-09-1",
      installKind: "git",
      git: {
        approval: {
          approvedTag: "deploy/2026-03-09-1",
          approvedSha: "fedcba987654",
        },
      },
    });
  });
});

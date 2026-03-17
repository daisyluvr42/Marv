import fs from "node:fs/promises";
import path from "node:path";
import { resolveStateDir } from "../../core/config/paths.js";
import type { CommandRunner } from "./update-steps.js";

export type CiStatusResult = {
  sha: string;
  passed: boolean;
  source: "github-api" | "local-cache" | "unavailable";
};

type CiCacheEntry = {
  sha: string;
  passed: boolean;
  checkedAt: string;
};

type CiCacheFile = {
  entries: Record<string, CiCacheEntry>;
};

const CACHE_FILENAME = "ci-status-cache.json";
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const MAX_CACHE_ENTRIES = 50;
const FETCH_TIMEOUT_MS = 5000;

// ── Cache ─────────────────────────────────────────────────────────────

function resolveCachePath(): string {
  return path.join(resolveStateDir(), CACHE_FILENAME);
}

async function readCache(): Promise<CiCacheFile> {
  try {
    const raw = await fs.readFile(resolveCachePath(), "utf-8");
    const parsed = JSON.parse(raw) as CiCacheFile;
    return parsed?.entries ? parsed : { entries: {} };
  } catch {
    return { entries: {} };
  }
}

async function writeCache(cache: CiCacheFile): Promise<void> {
  const cachePath = resolveCachePath();
  await fs.mkdir(path.dirname(cachePath), { recursive: true });
  // Evict old entries beyond limit
  const entries = Object.entries(cache.entries)
    .toSorted(([, a], [, b]) => Date.parse(b.checkedAt) - Date.parse(a.checkedAt))
    .slice(0, MAX_CACHE_ENTRIES);
  const trimmed: CiCacheFile = { entries: Object.fromEntries(entries) };
  await fs.writeFile(cachePath, JSON.stringify(trimmed, null, 2), "utf-8");
}

function getCachedStatus(cache: CiCacheFile, sha: string): CiStatusResult | null {
  const entry = cache.entries[sha];
  if (!entry) {
    return null;
  }
  const age = Date.now() - Date.parse(entry.checkedAt);
  if (age > CACHE_TTL_MS) {
    return null;
  }
  return { sha, passed: entry.passed, source: "local-cache" };
}

// ── GitHub API ────────────────────────────────────────────────────────

/** Extract owner/repo from a git remote URL. */
function parseGitHubRemote(remoteUrl: string): { owner: string; repo: string } | null {
  // HTTPS: https://github.com/owner/repo.git
  const httpsMatch = remoteUrl.match(/github\.com[/:]([^/]+)\/([^/.]+)/);
  if (httpsMatch) {
    return { owner: httpsMatch[1], repo: httpsMatch[2] };
  }
  return null;
}

async function getRemoteUrl(
  runCommand: CommandRunner,
  root: string,
  timeoutMs: number,
): Promise<string | null> {
  const res = await runCommand(["git", "-C", root, "remote", "get-url", "origin"], {
    timeoutMs,
  }).catch(() => null);
  if (!res || res.code !== 0) {
    return null;
  }
  return res.stdout.trim() || null;
}

type GitHubCombinedStatus = {
  state: string; // "success" | "failure" | "pending" | "error"
};

type GitHubCheckRun = {
  conclusion: string | null;
  status: string;
};

type GitHubCheckRunsResponse = {
  check_runs: GitHubCheckRun[];
};

async function fetchGitHubCiStatus(
  owner: string,
  repo: string,
  sha: string,
): Promise<boolean | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    // Check combined status (covers third-party CI like external status checks)
    const statusUrl = `https://api.github.com/repos/${owner}/${repo}/commits/${sha}/status`;
    const statusRes = await fetch(statusUrl, {
      headers: { Accept: "application/vnd.github+json" },
      signal: controller.signal,
    });

    if (statusRes.ok) {
      const data = (await statusRes.json()) as GitHubCombinedStatus;
      if (data.state === "success") {
        return true;
      }
      if (data.state === "failure" || data.state === "error") {
        return false;
      }
    }

    // Also check GitHub Actions check-runs
    const checksUrl = `https://api.github.com/repos/${owner}/${repo}/commits/${sha}/check-runs`;
    const checksRes = await fetch(checksUrl, {
      headers: { Accept: "application/vnd.github+json" },
      signal: controller.signal,
    });

    if (checksRes.ok) {
      const data = (await checksRes.json()) as GitHubCheckRunsResponse;
      const runs = data.check_runs ?? [];
      if (runs.length === 0) {
        return null; // No CI configured
      }
      const allCompleted = runs.every((r) => r.status === "completed");
      if (!allCompleted) {
        return null; // Still running
      }
      const allPassed = runs.every(
        (r) =>
          r.conclusion === "success" || r.conclusion === "skipped" || r.conclusion === "neutral",
      );
      return allPassed;
    }

    return null;
  } catch {
    return null; // Network error, rate limit, etc.
  } finally {
    clearTimeout(timeout);
  }
}

// ── Public API ────────────────────────────────────────────────────────

/**
 * Check CI status for a given commit SHA.
 *
 * Returns cached results when available. Falls back to GitHub API.
 * Gracefully returns `unavailable` if API is unreachable.
 */
export async function checkCiStatus(params: {
  runCommand: CommandRunner;
  root: string;
  sha: string;
  timeoutMs: number;
}): Promise<CiStatusResult> {
  const { runCommand, root, sha, timeoutMs } = params;

  // Check local cache first
  const cache = await readCache();
  const cached = getCachedStatus(cache, sha);
  if (cached) {
    return cached;
  }

  // Resolve GitHub remote
  const remoteUrl = await getRemoteUrl(runCommand, root, timeoutMs);
  if (!remoteUrl) {
    return { sha, passed: false, source: "unavailable" };
  }

  const remote = parseGitHubRemote(remoteUrl);
  if (!remote) {
    return { sha, passed: false, source: "unavailable" };
  }

  // Query GitHub API
  const passed = await fetchGitHubCiStatus(remote.owner, remote.repo, sha);
  if (passed == null) {
    return { sha, passed: false, source: "unavailable" };
  }

  // Cache the result
  cache.entries[sha] = {
    sha,
    passed,
    checkedAt: new Date().toISOString(),
  };
  await writeCache(cache).catch(() => {});

  return { sha, passed, source: "github-api" };
}

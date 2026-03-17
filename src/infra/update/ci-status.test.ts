import { describe, expect, it, vi } from "vitest";
import { checkCiStatus } from "./ci-status.js";
import type { CommandRunner } from "./update-steps.js";

function mockRunner(responses: Record<string, { stdout: string; code: number }>): CommandRunner {
  return async (argv) => {
    const key = argv.join(" ");
    for (const [pattern, response] of Object.entries(responses)) {
      if (key.includes(pattern)) {
        return { stdout: response.stdout, stderr: "", code: response.code };
      }
    }
    return { stdout: "", stderr: "not found", code: 1 };
  };
}

describe("checkCiStatus", () => {
  it("returns unavailable when remote cannot be resolved", async () => {
    const runner = mockRunner({
      "remote get-url": { stdout: "", code: 1 },
    });
    const result = await checkCiStatus({
      runCommand: runner,
      root: "/tmp/fake",
      sha: "abc123",
      timeoutMs: 5000,
    });
    expect(result.source).toBe("unavailable");
    expect(result.passed).toBe(false);
  });

  it("returns unavailable for non-github remotes", async () => {
    const runner = mockRunner({
      "remote get-url": { stdout: "https://gitlab.com/user/repo.git\n", code: 0 },
    });
    const result = await checkCiStatus({
      runCommand: runner,
      root: "/tmp/fake",
      sha: "abc123",
      timeoutMs: 5000,
    });
    expect(result.source).toBe("unavailable");
  });

  it("resolves github remote URL correctly", async () => {
    // Mock fetch to avoid real network calls
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ state: "success" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const runner = mockRunner({
      "remote get-url": { stdout: "https://github.com/owner/repo.git\n", code: 0 },
    });

    const result = await checkCiStatus({
      runCommand: runner,
      root: "/tmp/fake",
      sha: "abc12345",
      timeoutMs: 5000,
    });

    expect(result.passed).toBe(true);
    expect(result.source).toBe("github-api");
    expect(fetchSpy).toHaveBeenCalled();

    fetchSpy.mockRestore();
  });

  it("handles fetch failure gracefully", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("network error"));

    const runner = mockRunner({
      "remote get-url": { stdout: "https://github.com/owner/repo.git\n", code: 0 },
    });

    const result = await checkCiStatus({
      runCommand: runner,
      root: "/tmp/fake",
      sha: "def67890",
      timeoutMs: 5000,
    });

    expect(result.source).toBe("unavailable");
    expect(result.passed).toBe(false);

    fetchSpy.mockRestore();
  });
});

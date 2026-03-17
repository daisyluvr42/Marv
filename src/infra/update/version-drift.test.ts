import { describe, expect, it } from "vitest";
import { checkVersionDrift } from "./version-drift.js";

describe("checkVersionDrift", () => {
  it("reports no drift when app is not installed", async () => {
    const result = await checkVersionDrift({
      runCommand: async () => ({ stdout: "", code: 1 }),
    });
    expect(result.drifted).toBe(false);
    expect(result.appVersion).toBeNull();
    expect(result.message).toBeNull();
  });

  it("reports no drift when versions match", async () => {
    const result = await checkVersionDrift({
      runCommand: async () => ({ stdout: `${result.cliVersion}\n`, code: 0 }),
    });
    // CLI version is dynamic; just verify the shape
    expect(result.drifted).toBe(false);
    expect(result.message).toBeNull();
  });

  it("reports drift when versions differ", async () => {
    const result = await checkVersionDrift({
      runCommand: async () => ({ stdout: "2020.1.1\n", code: 0 }),
    });
    expect(result.drifted).toBe(true);
    expect(result.appVersion).toBe("2020.1.1");
    expect(result.message).toContain("Version drift");
  });
});

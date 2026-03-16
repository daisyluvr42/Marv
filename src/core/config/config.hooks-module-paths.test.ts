import { describe, expect, it } from "vitest";
import { validateConfigObjectWithPlugins } from "./config.js";

describe("config hooks module paths", () => {
  it("rejects absolute hooks.mappings[].transform.module", () => {
    const res = validateConfigObjectWithPlugins({
      agents: { list: [{ id: "pi" }] },
      hooks: {
        mappings: [
          {
            match: { path: "custom" },
            action: "agent",
            transform: { module: "/tmp/transform.mjs" },
          },
        ],
      },
    });
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.issues.some((iss) => iss.path === "hooks.mappings.0.transform.module")).toBe(true);
    }
  });

  it("rejects escaping hooks.mappings[].transform.module", () => {
    const res = validateConfigObjectWithPlugins({
      agents: { list: [{ id: "pi" }] },
      hooks: {
        mappings: [
          {
            match: { path: "custom" },
            action: "agent",
            transform: { module: "../escape.mjs" },
          },
        ],
      },
    });
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.issues.some((iss) => iss.path === "hooks.mappings.0.transform.module")).toBe(true);
    }
  });
});

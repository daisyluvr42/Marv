import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { MarvConfig } from "../core/config/config.js";
import { resolveStateDir } from "../core/config/paths.js";
import { buildAgentSummaries } from "./agents.js";

describe("agents helpers", () => {
  it("buildAgentSummaries only returns main in the main-only architecture", () => {
    const cfg: MarvConfig = {
      agents: {
        defaults: {
          name: "Marv",
          workspace: "/main-ws",
          model: { primary: "anthropic/claude" },
        },
      },
    };

    const summaries = buildAgentSummaries(cfg);
    expect(summaries).toHaveLength(1);
    const [main] = summaries;

    expect(main).toBeTruthy();
    expect(main?.id).toBe("main");
    expect(main?.name).toBe("Marv");
    expect(main?.workspace).toBe(path.resolve("/main-ws"));
    expect(main?.model).toBe("anthropic/claude");
    expect(main?.agentDir).toBe(
      path.join(resolveStateDir(process.env, os.homedir), "agents", "main", "agent"),
    );
    expect(main?.isDefault).toBe(true);
  });
});

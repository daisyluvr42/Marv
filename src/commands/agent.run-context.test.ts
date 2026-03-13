import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { withTempHome as withTempHomeBase } from "../../test/helpers/temp-home.js";

vi.mock("../agents/runner/pi-embedded.js", () => ({
  abortEmbeddedPiRun: vi.fn().mockReturnValue(false),
  runEmbeddedPiAgent: vi.fn().mockResolvedValue({
    payloads: [{ text: "ok" }],
    meta: {
      durationMs: 5,
      agentMeta: { sessionId: "s", provider: "p", model: "m" },
    },
  }),
  resolveEmbeddedSessionLane: (key: string) => `session:${key.trim() || "main"}`,
}));
vi.mock("../agents/model/model-catalog.js", () => ({
  loadModelCatalog: vi.fn().mockResolvedValue([]),
}));

import * as configModule from "../core/config/config.js";
import { AGENT_EVENT_REQUIRED_FIELDS, onAgentEvent } from "../infra/agent-events.js";
import type { RuntimeEnv } from "../runtime.js";
import { agentCommand } from "./agent.js";

const runtime: RuntimeEnv = {
  log: vi.fn(),
  error: vi.fn(),
  exit: vi.fn(() => {
    throw new Error("exit");
  }),
};

const configSpy = vi.spyOn(configModule, "loadConfig");

async function withTempHome<T>(fn: (home: string) => Promise<T>): Promise<T> {
  return withTempHomeBase(fn, { prefix: "marv-agent-run-context-" });
}

describe("agentCommand run context", () => {
  it("emits user-run context on the shared agent event bus", async () => {
    await withTempHome(async (home) => {
      const store = path.join(home, "sessions.json");
      configSpy.mockReturnValue({
        agents: {
          defaults: {
            model: { primary: "anthropic/claude-opus-4-5" },
            models: { "anthropic/claude-opus-4-5": {} },
            workspace: path.join(home, "marv"),
          },
        },
        session: { store, mainKey: "main" },
      });

      const seenContexts: Array<{
        requiredFields: string[];
        sessionKey?: string;
        context?: { sessionKey?: string; runModeKind?: string; verboseLevel?: string };
      }> = [];
      const stop = onAgentEvent((evt) => {
        if (evt.context) {
          seenContexts.push({
            requiredFields: AGENT_EVENT_REQUIRED_FIELDS.filter((field) =>
              Object.prototype.hasOwnProperty.call(evt, field),
            ),
            sessionKey: evt.sessionKey,
            context: evt.context as {
              sessionKey?: string;
              runModeKind?: string;
              verboseLevel?: string;
            },
          });
        }
      });

      await agentCommand({ message: "hi", to: "+1555", verbose: "on" }, runtime);
      stop();

      expect(
        seenContexts.some(
          (evt) =>
            evt.requiredFields.length === AGENT_EVENT_REQUIRED_FIELDS.length &&
            evt.context?.runModeKind === "user" &&
            evt.context?.verboseLevel === "on" &&
            evt.context?.sessionKey === evt.sessionKey &&
            typeof evt.sessionKey === "string" &&
            evt.sessionKey.length > 0,
        ),
      ).toBe(true);
    });
  });
});

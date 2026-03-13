import { beforeAll, describe, expect, it } from "vitest";
import { resolveControlCommandGate } from "../channels/command-gating.js";
import {
  getRunEmbeddedPiAgentMock,
  installTriggerHandlingE2eTestHooks,
  makeCfg,
  withTempHome,
} from "./reply.triggers.trigger-handling.test-harness.js";

let getReplyFromConfig: typeof import("./reply.js").getReplyFromConfig;
beforeAll(async () => {
  ({ getReplyFromConfig } = await import("./reply.js"));
});

installTriggerHandlingE2eTestHooks();

function mockEmbeddedOk() {
  const runEmbeddedPiAgentMock = getRunEmbeddedPiAgentMock();
  runEmbeddedPiAgentMock.mockResolvedValue({
    payloads: [{ text: "ok" }],
    meta: {
      durationMs: 1,
      agentMeta: { sessionId: "s", provider: "p", model: "m" },
    },
  });
  return runEmbeddedPiAgentMock;
}

describe("trigger handling channel policy seam", () => {
  it("drops exact control commands when the shared gate blocks them", async () => {
    await withTempHome(async (home) => {
      const runEmbeddedPiAgentMock = getRunEmbeddedPiAgentMock();
      const commandGate = resolveControlCommandGate({
        useAccessGroups: true,
        authorizers: [{ configured: true, allowed: false }],
        allowTextCommands: true,
        hasControlCommand: true,
      });

      expect(commandGate).toEqual({ commandAuthorized: false, shouldBlock: true });

      const res = await getReplyFromConfig(
        {
          Body: "/status",
          From: "+2001",
          To: "+2000",
          Provider: "whatsapp",
          SenderE164: "+2001",
          CommandAuthorized: commandGate.commandAuthorized,
        },
        {},
        makeCfg(home),
      );

      expect(res).toBeUndefined();
      expect(runEmbeddedPiAgentMock).not.toHaveBeenCalled();
    });
  });

  it("keeps inline control text in the prompt when the shared gate does not block", async () => {
    await withTempHome(async (home) => {
      const runEmbeddedPiAgentMock = mockEmbeddedOk();
      const commandGate = resolveControlCommandGate({
        useAccessGroups: true,
        authorizers: [{ configured: true, allowed: false }],
        allowTextCommands: false,
        hasControlCommand: true,
      });

      expect(commandGate).toEqual({ commandAuthorized: false, shouldBlock: false });

      const res = await getReplyFromConfig(
        {
          Body: "please /status now",
          From: "+2001",
          To: "+2000",
          Provider: "whatsapp",
          SenderE164: "+2001",
          CommandAuthorized: commandGate.commandAuthorized,
        },
        {},
        makeCfg(home),
      );

      const text = Array.isArray(res) ? res[0]?.text : res?.text;
      expect(text).toBe("ok");
      expect(runEmbeddedPiAgentMock).toHaveBeenCalledOnce();
      const prompt = runEmbeddedPiAgentMock.mock.calls[0]?.[0]?.prompt ?? "";
      expect(prompt).toContain("/status");
    });
  });
});

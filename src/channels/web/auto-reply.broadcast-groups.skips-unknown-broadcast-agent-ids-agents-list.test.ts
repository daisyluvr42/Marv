import "./test-helpers.js";
import { describe, expect, it } from "vitest";
import type { MarvConfig } from "../../core/config/config.js";
import { sendWebDirectInboundAndCollectSessionKeys } from "./auto-reply.broadcast-groups.test-harness.js";
import {
  installWebAutoReplyTestHomeHooks,
  installWebAutoReplyUnitTestHooks,
  resetLoadConfigMock,
  setLoadConfigMock,
} from "./auto-reply.test-harness.js";

installWebAutoReplyTestHomeHooks();

describe("broadcast groups", () => {
  installWebAutoReplyUnitTestHooks();

  it("resolves broadcast to main agent when only main agent exists", async () => {
    setLoadConfigMock({
      channels: { whatsapp: { allowFrom: ["*"] } },
      agents: {
        defaults: { maxConcurrent: 10 },
      },
      broadcast: {
        "+1000": ["main", "missing"],
      },
    } satisfies MarvConfig);

    const { seen, resolver } = await sendWebDirectInboundAndCollectSessionKeys();

    expect(resolver).toHaveBeenCalledTimes(1);
    expect(seen[0]).toContain("agent:main:");
    resetLoadConfigMock();
  });
});

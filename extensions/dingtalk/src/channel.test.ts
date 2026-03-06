import type { MarvConfig } from "marv/plugin-sdk";
import { describe, expect, it, vi } from "vitest";

const probeDingTalkMock = vi.hoisted(() => vi.fn());

vi.mock("./api.js", async () => {
  const actual = await vi.importActual<typeof import("./api.js")>("./api.js");
  return {
    ...actual,
    probeDingTalk: probeDingTalkMock,
  };
});

import { dingtalkPlugin } from "./channel.js";

describe("dingtalkPlugin.status.probeAccount", () => {
  it("uses current account credentials for multi-account config", async () => {
    const cfg = {
      channels: {
        dingtalk: {
          enabled: true,
          accounts: {
            main: {
              clientId: "dt_main",
              clientSecret: "secret_main",
              robotCode: "robot_main",
              enabled: true,
            },
          },
        },
      },
    } as MarvConfig;

    const account = dingtalkPlugin.config.resolveAccount(cfg, "main");
    probeDingTalkMock.mockResolvedValueOnce({ ok: true, clientId: "dt_main" });

    const result = await dingtalkPlugin.status?.probeAccount?.({
      account,
      timeoutMs: 1_000,
      cfg,
    });

    expect(probeDingTalkMock).toHaveBeenCalledWith(
      expect.objectContaining({
        accountId: "main",
        clientId: "dt_main",
        clientSecret: "secret_main",
        robotCode: "robot_main",
      }),
    );
    expect(result).toMatchObject({ ok: true, clientId: "dt_main" });
  });
});

import { afterEach, describe, expect, it, vi } from "vitest";
import { probeDingTalk } from "./api.js";

describe("probeDingTalk", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fails early when credentials are missing", async () => {
    const result = await probeDingTalk({
      accountId: "main",
      enabled: true,
      configured: false,
      config: {},
    });
    expect(result).toEqual({
      ok: false,
      accountId: "main",
      clientId: undefined,
      robotCode: undefined,
      stage: "credentials",
      error: "missing credentials (clientId, clientSecret)",
    });
  });

  it("probes token endpoint with account credentials", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        text: async () => JSON.stringify({ accessToken: "tok", expireIn: 7200 }),
      }),
    );

    const result = await probeDingTalk({
      accountId: "main",
      enabled: true,
      configured: true,
      clientId: "cli",
      clientSecret: "sec",
      robotCode: "robot",
      config: {},
    });

    expect(result).toEqual({
      ok: true,
      accountId: "main",
      clientId: "cli",
      robotCode: "robot",
      stage: "token",
    });
    expect(fetch).toHaveBeenCalledWith(
      "https://api.dingtalk.com/v1.0/oauth2/accessToken",
      expect.objectContaining({
        method: "POST",
      }),
    );
  });
});

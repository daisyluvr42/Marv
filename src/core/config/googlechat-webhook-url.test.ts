import { describe, expect, it } from "vitest";
import { validateConfigObject } from "./config.js";

describe("Google Chat webhook config", () => {
  it("accepts http(s) webhook URLs", () => {
    const res = validateConfigObject({
      channels: {
        googlechat: {
          webhookUrl: "https://example.com/googlechat-webhook",
        },
      },
    });
    expect(res.ok).toBe(true);
  });

  it("rejects non-http webhook URLs", () => {
    const res = validateConfigObject({
      channels: {
        googlechat: {
          webhookUrl: "ftp://example.com/googlechat-webhook",
        },
      },
    });
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.issues.some((issue) => issue.path === "channels.googlechat.webhookUrl")).toBe(
        true,
      );
    }
  });

  it("rejects non-http account webhook URLs", () => {
    const res = validateConfigObject({
      channels: {
        googlechat: {
          accounts: {
            ops: {
              webhookUrl: "ftp://example.com/googlechat-webhook",
            },
          },
        },
      },
    });
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(
        res.issues.some((issue) => issue.path === "channels.googlechat.accounts.ops.webhookUrl"),
      ).toBe(true);
    }
  });
});

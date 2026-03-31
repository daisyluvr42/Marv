import { describe, expect, it } from "vitest";
import { logGatewayStartup } from "./server-startup-log.js";

describe("logGatewayStartup", () => {
  it("logs version and build identity before the listen address", () => {
    const lines: string[] = [];

    logGatewayStartup({
      cfg: {},
      bindHost: "127.0.0.1",
      port: 4242,
      log: {
        info: (message) => lines.push(message),
      },
      isNixMode: false,
    });

    const buildLine = lines.find((line) => line.startsWith("build: version "));
    expect(buildLine).toBeDefined();
    expect(buildLine).toContain(`exec ${process.execPath}`);
    expect(buildLine).toContain(`entry ${process.argv[1] || process.execPath}`);

    const buildIndex = lines.findIndex((line) => line.startsWith("build: version "));
    const listenIndex = lines.findIndex((line) => line.startsWith("listening on "));
    expect(buildIndex).toBeGreaterThanOrEqual(0);
    expect(listenIndex).toBeGreaterThan(buildIndex);
  });
});

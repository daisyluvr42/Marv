import { Command } from "commander";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { registerBrowserDebugCommands } from "./browser-cli-debug.js";
import type { BrowserParentOpts } from "./browser-cli-shared.js";

const mocks = vi.hoisted(() => ({
  callBrowserRequest: vi.fn(async () => ({ errors: [] })),
  runtime: {
    log: vi.fn(),
    error: vi.fn(),
    exit: vi.fn(),
  },
}));

vi.mock("./browser-cli-shared.js", () => ({
  callBrowserRequest: mocks.callBrowserRequest,
}));

vi.mock("../runtime.js", () => ({
  defaultRuntime: mocks.runtime,
}));

describe("browser debug commands", () => {
  beforeEach(() => {
    mocks.callBrowserRequest.mockReset();
    mocks.runtime.log.mockClear();
    mocks.runtime.error.mockClear();
    mocks.runtime.exit.mockClear();
  });

  it("forwards parent browser profile and request flags to `browser errors`", async () => {
    mocks.callBrowserRequest.mockResolvedValueOnce({
      errors: [{ timestamp: "2026-03-13T10:00:00Z", name: "TypeError", message: "boom" }],
    });

    const program = new Command();
    const browser = program
      .command("browser")
      .option("--browser-profile <name>", "Browser profile")
      .option("--json", "Output JSON", false);
    const parentOpts = (cmd: Command) => cmd.parent?.opts?.() as BrowserParentOpts;
    registerBrowserDebugCommands(browser, parentOpts);

    await program.parseAsync(
      ["browser", "--browser-profile", "work", "errors", "--clear", "--target-id", "tab-7"],
      { from: "user" },
    );

    const call = mocks.callBrowserRequest.mock.calls.at(-1);
    expect(call).toBeDefined();
    if (!call) {
      throw new Error("expected browser request call");
    }
    const request = call[1] as { path?: string; query?: Record<string, string> };
    expect(request.path).toBe("/errors");
    expect(request.query).toMatchObject({
      profile: "work",
      clear: true,
      targetId: "tab-7",
    });
    expect(mocks.runtime.log).toHaveBeenCalledWith("2026-03-13T10:00:00Z TypeError: boom");
  });

  it("keeps `browser requests --json` machine-readable", async () => {
    mocks.callBrowserRequest.mockResolvedValueOnce({
      requests: [
        {
          timestamp: "2026-03-13T10:00:00Z",
          method: "GET",
          status: 200,
          ok: true,
          url: "https://example.com/api",
        },
      ],
    });

    const program = new Command();
    const browser = program
      .command("browser")
      .option("--browser-profile <name>", "Browser profile")
      .option("--json", "Output JSON", false);
    const parentOpts = (cmd: Command) => cmd.parent?.opts?.() as BrowserParentOpts;
    registerBrowserDebugCommands(browser, parentOpts);

    await program.parseAsync(
      ["browser", "--browser-profile", "work", "--json", "requests", "--filter", "api"],
      { from: "user" },
    );

    const call = mocks.callBrowserRequest.mock.calls.at(-1);
    expect(call).toBeDefined();
    if (!call) {
      throw new Error("expected browser request call");
    }
    const request = call[1] as { path?: string; query?: Record<string, string> };
    expect(request.path).toBe("/requests");
    expect(request.query).toMatchObject({
      profile: "work",
      filter: "api",
    });
    expect(mocks.runtime.log).toHaveBeenCalledWith(
      JSON.stringify(
        {
          requests: [
            {
              timestamp: "2026-03-13T10:00:00Z",
              method: "GET",
              status: 200,
              ok: true,
              url: "https://example.com/api",
            },
          ],
        },
        null,
        2,
      ),
    );
  });

  it("formats `browser requests` for humans with resource type and failures", async () => {
    mocks.callBrowserRequest.mockResolvedValueOnce({
      requests: [
        {
          timestamp: "2026-03-13T10:00:00Z",
          method: "POST",
          status: 503,
          ok: false,
          url: "https://example.com/api",
          resourceType: "xhr",
          failureText: "upstream timeout",
        },
      ],
    });

    const program = new Command();
    const browser = program
      .command("browser")
      .option("--browser-profile <name>", "Browser profile")
      .option("--json", "Output JSON", false);
    const parentOpts = (cmd: Command) => cmd.parent?.opts?.() as BrowserParentOpts;
    registerBrowserDebugCommands(browser, parentOpts);

    await program.parseAsync(["browser", "requests"], { from: "user" });

    expect(mocks.runtime.log).toHaveBeenCalledWith(
      "2026-03-13T10:00:00Z POST 503 fail [xhr] https://example.com/api (upstream timeout)",
    );
  });

  it("shows a concise empty state for `browser errors` in human mode", async () => {
    mocks.callBrowserRequest.mockResolvedValueOnce({ errors: [] });

    const program = new Command();
    const browser = program
      .command("browser")
      .option("--browser-profile <name>", "Browser profile")
      .option("--json", "Output JSON", false);
    const parentOpts = (cmd: Command) => cmd.parent?.opts?.() as BrowserParentOpts;
    registerBrowserDebugCommands(browser, parentOpts);

    await program.parseAsync(["browser", "errors"], { from: "user" });

    expect(mocks.runtime.log).toHaveBeenCalledWith("No page errors.");
  });
});

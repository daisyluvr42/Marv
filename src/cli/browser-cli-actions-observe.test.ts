import { Command } from "commander";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { registerBrowserActionObserveCommands } from "./browser-cli-actions-observe.js";
import type { BrowserParentOpts } from "./browser-cli-shared.js";

const mocks = vi.hoisted(() => ({
  callBrowserRequest: vi.fn(async () => ({ messages: [] })),
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

describe("browser observe commands", () => {
  beforeEach(() => {
    mocks.callBrowserRequest.mockReset();
    mocks.runtime.log.mockClear();
    mocks.runtime.error.mockClear();
    mocks.runtime.exit.mockClear();
  });

  it("passes browser profile and timeout controls to `browser responsebody`", async () => {
    mocks.callBrowserRequest.mockResolvedValueOnce({
      response: { body: '{"ok":true}' },
    });

    const program = new Command();
    const browser = program
      .command("browser")
      .option("--browser-profile <name>", "Browser profile")
      .option("--json", "Output JSON", false);
    const parentOpts = (cmd: Command) => cmd.parent?.opts?.() as BrowserParentOpts;
    registerBrowserActionObserveCommands(browser, parentOpts);

    await program.parseAsync(
      [
        "browser",
        "--browser-profile",
        "lab",
        "responsebody",
        "**/api",
        "--target-id",
        "tab-9",
        "--timeout-ms",
        "3210",
        "--max-chars",
        "1234",
      ],
      { from: "user" },
    );

    const call = mocks.callBrowserRequest.mock.calls.at(-1);
    expect(call).toBeDefined();
    if (!call) {
      throw new Error("expected browser request call");
    }
    const request = call[1] as {
      path?: string;
      query?: Record<string, string>;
      body?: { url?: string; targetId?: string; timeoutMs?: number; maxChars?: number };
    };
    const extra = call[2] as { timeoutMs?: number } | undefined;
    expect(request.path).toBe("/response/body");
    expect(request.query).toMatchObject({ profile: "lab" });
    expect(request.body).toMatchObject({
      url: "**/api",
      targetId: "tab-9",
      timeoutMs: 3210,
      maxChars: 1234,
    });
    expect(extra?.timeoutMs).toBe(3210);
    expect(mocks.runtime.log).toHaveBeenCalledWith('{"ok":true}');
  });

  it("keeps `browser console --json` machine-readable", async () => {
    mocks.callBrowserRequest.mockResolvedValueOnce({
      messages: [{ level: "warn", text: "careful" }],
    });

    const program = new Command();
    const browser = program
      .command("browser")
      .option("--browser-profile <name>", "Browser profile")
      .option("--json", "Output JSON", false);
    const parentOpts = (cmd: Command) => cmd.parent?.opts?.() as BrowserParentOpts;
    registerBrowserActionObserveCommands(browser, parentOpts);

    await program.parseAsync(
      ["browser", "--browser-profile", "lab", "--json", "console", "--level", "warn"],
      { from: "user" },
    );

    const call = mocks.callBrowserRequest.mock.calls.at(-1);
    expect(call).toBeDefined();
    if (!call) {
      throw new Error("expected browser request call");
    }
    const request = call[1] as { path?: string; query?: Record<string, string> };
    expect(request.path).toBe("/console");
    expect(request.query).toMatchObject({
      profile: "lab",
      level: "warn",
    });
    expect(mocks.runtime.log).toHaveBeenCalledWith(
      JSON.stringify(
        {
          messages: [{ level: "warn", text: "careful" }],
        },
        null,
        2,
      ),
    );
  });

  it("formats `browser console` for humans without losing location context", async () => {
    mocks.callBrowserRequest.mockResolvedValueOnce({
      messages: [
        {
          timestamp: "2026-03-13T10:00:00Z",
          type: "warn",
          text: "careful",
          location: {
            url: "https://example.com/app",
            lineNumber: 12,
            columnNumber: 4,
          },
        },
      ],
    });

    const program = new Command();
    const browser = program
      .command("browser")
      .option("--browser-profile <name>", "Browser profile")
      .option("--json", "Output JSON", false);
    const parentOpts = (cmd: Command) => cmd.parent?.opts?.() as BrowserParentOpts;
    registerBrowserActionObserveCommands(browser, parentOpts);

    await program.parseAsync(["browser", "console"], { from: "user" });

    expect(mocks.runtime.log).toHaveBeenCalledWith(
      "2026-03-13T10:00:00Z warn careful @ https://example.com/app:12:4",
    );
  });

  it("shows a concise empty-state for `browser console` in human mode", async () => {
    mocks.callBrowserRequest.mockResolvedValueOnce({ messages: [] });

    const program = new Command();
    const browser = program
      .command("browser")
      .option("--browser-profile <name>", "Browser profile")
      .option("--json", "Output JSON", false);
    const parentOpts = (cmd: Command) => cmd.parent?.opts?.() as BrowserParentOpts;
    registerBrowserActionObserveCommands(browser, parentOpts);

    await program.parseAsync(["browser", "console"], { from: "user" });

    expect(mocks.runtime.log).toHaveBeenCalledWith("No console messages.");
  });
});

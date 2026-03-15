import { Command } from "commander";
import { describe, expect, it } from "vitest";
import {
  commandMatchesPrimary,
  rewriteUpdateFlagArgv,
  shouldEnsureCliPath,
  shouldRegisterPrimarySubcommand,
  shouldSkipPluginCommandRegistration,
} from "./run-main.js";

describe("rewriteUpdateFlagArgv", () => {
  it("leaves argv unchanged when --update is absent", () => {
    const argv = ["node", "entry.js", "status"];
    expect(rewriteUpdateFlagArgv(argv)).toBe(argv);
  });

  it("rewrites --update into the update command", () => {
    expect(rewriteUpdateFlagArgv(["node", "entry.js", "--update"])).toEqual([
      "node",
      "entry.js",
      "update",
    ]);
  });

  it("preserves global flags that appear before --update", () => {
    expect(rewriteUpdateFlagArgv(["node", "entry.js", "--profile", "p", "--update"])).toEqual([
      "node",
      "entry.js",
      "--profile",
      "p",
      "update",
    ]);
  });

  it("keeps update options after the rewritten command", () => {
    expect(rewriteUpdateFlagArgv(["node", "entry.js", "--update", "--json"])).toEqual([
      "node",
      "entry.js",
      "update",
      "--json",
    ]);
  });
});

describe("shouldRegisterPrimarySubcommand", () => {
  it("skips eager primary registration for help/version invocations", () => {
    expect(shouldRegisterPrimarySubcommand(["node", "marv", "status", "--help"])).toBe(false);
    expect(shouldRegisterPrimarySubcommand(["node", "marv", "-V"])).toBe(false);
  });

  it("keeps eager primary registration for regular command runs", () => {
    expect(shouldRegisterPrimarySubcommand(["node", "marv", "status"])).toBe(true);
  });
});

describe("shouldSkipPluginCommandRegistration", () => {
  it("skips plugin registration for root help/version", () => {
    expect(
      shouldSkipPluginCommandRegistration({
        argv: ["node", "marv", "--help"],
        primary: null,
        hasBuiltinPrimary: false,
      }),
    ).toBe(true);
  });

  it("skips plugin registration for builtin subcommand help", () => {
    expect(
      shouldSkipPluginCommandRegistration({
        argv: ["node", "marv", "config", "--help"],
        primary: "config",
        hasBuiltinPrimary: true,
      }),
    ).toBe(true);
  });

  it("skips plugin registration for builtin command runs", () => {
    expect(
      shouldSkipPluginCommandRegistration({
        argv: ["node", "marv", "sessions", "--json"],
        primary: "sessions",
        hasBuiltinPrimary: true,
      }),
    ).toBe(true);
  });

  it("keeps plugin registration for non-builtin help", () => {
    expect(
      shouldSkipPluginCommandRegistration({
        argv: ["node", "marv", "voicecall", "--help"],
        primary: "voicecall",
        hasBuiltinPrimary: false,
      }),
    ).toBe(false);
  });

  it("keeps plugin registration for non-builtin command runs", () => {
    expect(
      shouldSkipPluginCommandRegistration({
        argv: ["node", "marv", "voicecall", "status"],
        primary: "voicecall",
        hasBuiltinPrimary: false,
      }),
    ).toBe(false);
  });
});

describe("shouldEnsureCliPath", () => {
  it("skips path bootstrap for help/version invocations", () => {
    expect(shouldEnsureCliPath(["node", "marv", "--help"])).toBe(false);
    expect(shouldEnsureCliPath(["node", "marv", "-V"])).toBe(false);
  });

  it("skips path bootstrap for read-only fast paths", () => {
    expect(shouldEnsureCliPath(["node", "marv", "status"])).toBe(false);
    expect(shouldEnsureCliPath(["node", "marv", "sessions", "--json"])).toBe(false);
    expect(shouldEnsureCliPath(["node", "marv", "config", "get", "update"])).toBe(false);
    expect(shouldEnsureCliPath(["node", "marv", "config", "validate"])).toBe(false);
    expect(shouldEnsureCliPath(["node", "marv", "models", "status", "--json"])).toBe(false);
    expect(shouldEnsureCliPath(["node", "marv", "memory", "status"])).toBe(false);
    expect(shouldEnsureCliPath(["node", "marv", "browser", "console"])).toBe(false);
    expect(shouldEnsureCliPath(["node", "marv", "system", "heartbeat", "last"])).toBe(false);
    expect(shouldEnsureCliPath(["node", "marv", "update", "status"])).toBe(false);
  });

  it("keeps path bootstrap for mutating or unknown commands", () => {
    expect(shouldEnsureCliPath(["node", "marv", "message", "send"])).toBe(true);
    expect(shouldEnsureCliPath(["node", "marv", "voicecall", "status"])).toBe(true);
  });
});

describe("commandMatchesPrimary", () => {
  it("treats commander aliases as builtin matches", () => {
    const program = new Command();
    const mem = program.command("mem").alias("memory");

    expect(commandMatchesPrimary(mem, "mem")).toBe(true);
    expect(commandMatchesPrimary(mem, "memory")).toBe(true);
    expect(commandMatchesPrimary(mem, "agent")).toBe(false);
  });
});

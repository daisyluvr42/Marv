import { describe, expect, it } from "vitest";
import {
  buildParseArgv,
  getFlagValue,
  getCommandPath,
  resolveCommandCliBootstrapFromPath,
  resolveCommandConfigValidityFromPath,
  getPrimaryCommand,
  getPositiveIntFlagValue,
  getVerboseFlag,
  hasHelpOrVersion,
  hasFlag,
  resolveCommandSideEffectFromPath,
  shouldMigrateState,
  shouldMigrateStateFromPath,
} from "./argv.js";

describe("argv helpers", () => {
  it("detects help/version flags", () => {
    expect(hasHelpOrVersion(["node", "marv", "--help"])).toBe(true);
    expect(hasHelpOrVersion(["node", "marv", "-V"])).toBe(true);
    expect(hasHelpOrVersion(["node", "marv", "status"])).toBe(false);
  });

  it("extracts command path ignoring flags and terminator", () => {
    expect(getCommandPath(["node", "marv", "status", "--json"], 2)).toEqual(["status"]);
    expect(getCommandPath(["node", "marv", "agents", "list"], 2)).toEqual(["agents", "list"]);
    expect(getCommandPath(["node", "marv", "system", "heartbeat", "last"], 3)).toEqual([
      "system",
      "heartbeat",
      "last",
    ]);
    expect(getCommandPath(["node", "marv", "status", "--", "ignored"], 2)).toEqual(["status"]);
  });

  it("returns primary command", () => {
    expect(getPrimaryCommand(["node", "marv", "agents", "list"])).toBe("agents");
    expect(getPrimaryCommand(["node", "marv"])).toBeNull();
  });

  it("parses boolean flags and ignores terminator", () => {
    expect(hasFlag(["node", "marv", "status", "--json"], "--json")).toBe(true);
    expect(hasFlag(["node", "marv", "--", "--json"], "--json")).toBe(false);
  });

  it("extracts flag values with equals and missing values", () => {
    expect(getFlagValue(["node", "marv", "status", "--timeout", "5000"], "--timeout")).toBe("5000");
    expect(getFlagValue(["node", "marv", "status", "--timeout=2500"], "--timeout")).toBe("2500");
    expect(getFlagValue(["node", "marv", "status", "--timeout"], "--timeout")).toBeNull();
    expect(getFlagValue(["node", "marv", "status", "--timeout", "--json"], "--timeout")).toBe(null);
    expect(getFlagValue(["node", "marv", "--", "--timeout=99"], "--timeout")).toBeUndefined();
  });

  it("parses verbose flags", () => {
    expect(getVerboseFlag(["node", "marv", "status", "--verbose"])).toBe(true);
    expect(getVerboseFlag(["node", "marv", "status", "--debug"])).toBe(false);
    expect(getVerboseFlag(["node", "marv", "status", "--debug"], { includeDebug: true })).toBe(
      true,
    );
  });

  it("parses positive integer flag values", () => {
    expect(getPositiveIntFlagValue(["node", "marv", "status"], "--timeout")).toBeUndefined();
    expect(
      getPositiveIntFlagValue(["node", "marv", "status", "--timeout"], "--timeout"),
    ).toBeNull();
    expect(
      getPositiveIntFlagValue(["node", "marv", "status", "--timeout", "5000"], "--timeout"),
    ).toBe(5000);
    expect(
      getPositiveIntFlagValue(["node", "marv", "status", "--timeout", "nope"], "--timeout"),
    ).toBeUndefined();
  });

  it("builds parse argv from raw args", () => {
    const nodeArgv = buildParseArgv({
      programName: "marv",
      rawArgs: ["node", "marv", "status"],
    });
    expect(nodeArgv).toEqual(["node", "marv", "status"]);

    const versionedNodeArgv = buildParseArgv({
      programName: "marv",
      rawArgs: ["node-22", "marv", "status"],
    });
    expect(versionedNodeArgv).toEqual(["node-22", "marv", "status"]);

    const versionedNodeWindowsArgv = buildParseArgv({
      programName: "marv",
      rawArgs: ["node-22.2.0.exe", "marv", "status"],
    });
    expect(versionedNodeWindowsArgv).toEqual(["node-22.2.0.exe", "marv", "status"]);

    const versionedNodePatchlessArgv = buildParseArgv({
      programName: "marv",
      rawArgs: ["node-22.2", "marv", "status"],
    });
    expect(versionedNodePatchlessArgv).toEqual(["node-22.2", "marv", "status"]);

    const versionedNodeWindowsPatchlessArgv = buildParseArgv({
      programName: "marv",
      rawArgs: ["node-22.2.exe", "marv", "status"],
    });
    expect(versionedNodeWindowsPatchlessArgv).toEqual(["node-22.2.exe", "marv", "status"]);

    const versionedNodeWithPathArgv = buildParseArgv({
      programName: "marv",
      rawArgs: ["/usr/bin/node-22.2.0", "marv", "status"],
    });
    expect(versionedNodeWithPathArgv).toEqual(["/usr/bin/node-22.2.0", "marv", "status"]);

    const nodejsArgv = buildParseArgv({
      programName: "marv",
      rawArgs: ["nodejs", "marv", "status"],
    });
    expect(nodejsArgv).toEqual(["nodejs", "marv", "status"]);

    const nonVersionedNodeArgv = buildParseArgv({
      programName: "marv",
      rawArgs: ["node-dev", "marv", "status"],
    });
    expect(nonVersionedNodeArgv).toEqual(["node", "marv", "node-dev", "marv", "status"]);

    const directArgv = buildParseArgv({
      programName: "marv",
      rawArgs: ["marv", "status"],
    });
    expect(directArgv).toEqual(["node", "marv", "status"]);

    const bunArgv = buildParseArgv({
      programName: "marv",
      rawArgs: ["bun", "src/entry.ts", "status"],
    });
    expect(bunArgv).toEqual(["bun", "src/entry.ts", "status"]);
  });

  it("builds parse argv from fallback args", () => {
    const fallbackArgv = buildParseArgv({
      programName: "marv",
      fallbackArgv: ["status"],
    });
    expect(fallbackArgv).toEqual(["node", "marv", "status"]);
  });

  it("decides when to migrate state", () => {
    expect(shouldMigrateState(["node", "marv", "status"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "health"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "sessions"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "logs"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "config", "get", "update"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "config", "validate"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "config", "unset", "update"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "gateway", "status"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "gateway", "probe"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "browser", "console"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "update", "status"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "daemon", "status"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "system", "presence"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "system", "heartbeat", "last"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "system", "heartbeat", "enable"])).toBe(true);
    expect(shouldMigrateState(["node", "marv", "models", "list"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "models", "status"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "memory", "status"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "agent", "--message", "hi"])).toBe(false);
    expect(shouldMigrateState(["node", "marv", "agents", "list"])).toBe(true);
    expect(shouldMigrateState(["node", "marv", "message", "send"])).toBe(true);
  });

  it("reuses command path for migrate state decisions", () => {
    expect(shouldMigrateStateFromPath(["status"])).toBe(false);
    expect(shouldMigrateStateFromPath(["logs"])).toBe(false);
    expect(shouldMigrateStateFromPath(["config", "get"])).toBe(false);
    expect(shouldMigrateStateFromPath(["config", "validate"])).toBe(false);
    expect(shouldMigrateStateFromPath(["gateway", "health"])).toBe(false);
    expect(shouldMigrateStateFromPath(["system", "heartbeat", "last"])).toBe(false);
    expect(shouldMigrateStateFromPath(["models", "status"])).toBe(false);
    expect(shouldMigrateStateFromPath(["agents", "list"])).toBe(true);
  });

  it("resolves side effects declaratively from command paths", () => {
    expect(resolveCommandSideEffectFromPath(["status"])).toBe("none");
    expect(resolveCommandSideEffectFromPath(["logs"])).toBe("none");
    expect(resolveCommandSideEffectFromPath(["config", "get"])).toBe("none");
    expect(resolveCommandSideEffectFromPath(["config", "validate"])).toBe("none");
    expect(resolveCommandSideEffectFromPath(["gateway", "discover"])).toBe("none");
    expect(resolveCommandSideEffectFromPath(["browser", "errors"])).toBe("none");
    expect(resolveCommandSideEffectFromPath(["system", "heartbeat", "last"])).toBe("none");
    expect(resolveCommandSideEffectFromPath(["system", "heartbeat", "enable"])).toBe(
      "state-migrate",
    );
    expect(resolveCommandSideEffectFromPath(["message", "send"])).toBe("state-migrate");
    expect(resolveCommandSideEffectFromPath([])).toBe("state-migrate");
  });

  it("resolves invalid-config allowance declaratively from command paths", () => {
    expect(resolveCommandConfigValidityFromPath(["status"])).toBe("allow-invalid");
    expect(resolveCommandConfigValidityFromPath(["config", "get"])).toBe("allow-invalid");
    expect(resolveCommandConfigValidityFromPath(["config", "validate"])).toBe("allow-invalid");
    expect(resolveCommandConfigValidityFromPath(["gateway", "probe"])).toBe("allow-invalid");
    expect(resolveCommandConfigValidityFromPath(["message", "send"])).toBe("require-valid");
    expect(resolveCommandConfigValidityFromPath([])).toBe("require-valid");
  });

  it("resolves CLI path bootstrap policy declaratively from command paths", () => {
    expect(resolveCommandCliBootstrapFromPath(["status"])).toBe("skip");
    expect(resolveCommandCliBootstrapFromPath(["config", "validate"])).toBe("skip");
    expect(resolveCommandCliBootstrapFromPath(["memory", "status"])).toBe("skip");
    expect(resolveCommandCliBootstrapFromPath(["browser", "console"])).toBe("skip");
    expect(resolveCommandCliBootstrapFromPath(["system", "heartbeat", "last"])).toBe("skip");
    expect(resolveCommandCliBootstrapFromPath(["system", "heartbeat", "enable"])).toBe("require");
    expect(resolveCommandCliBootstrapFromPath(["message", "send"])).toBe("require");
    expect(resolveCommandCliBootstrapFromPath([])).toBe("require");
  });
});

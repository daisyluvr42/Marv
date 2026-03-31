import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  resolveDefaultConfigCandidates,
  resolveConfigPathCandidate,
  resolveConfigPath,
  resolveOAuthDir,
  resolveOAuthPath,
  resolveStateDir,
} from "./paths.js";

describe("oauth paths", () => {
  it("prefers MARV_OAUTH_DIR over MARV_STATE_DIR", () => {
    const env = {
      MARV_OAUTH_DIR: "/custom/oauth",
      MARV_STATE_DIR: "/custom/state",
    } as NodeJS.ProcessEnv;

    expect(resolveOAuthDir(env, "/custom/state")).toBe(path.resolve("/custom/oauth"));
    expect(resolveOAuthPath(env, "/custom/state")).toBe(
      path.join(path.resolve("/custom/oauth"), "oauth.json"),
    );
  });

  it("derives oauth path from MARV_STATE_DIR when unset", () => {
    const env = {
      MARV_STATE_DIR: "/custom/state",
    } as NodeJS.ProcessEnv;

    expect(resolveOAuthDir(env, "/custom/state")).toBe(path.join("/custom/state", "credentials"));
    expect(resolveOAuthPath(env, "/custom/state")).toBe(
      path.join("/custom/state", "credentials", "oauth.json"),
    );
  });
});

describe("state + config path candidates", () => {
  it("uses MARV_STATE_DIR when set", () => {
    const env = {
      MARV_STATE_DIR: "/new/state",
    } as NodeJS.ProcessEnv;

    expect(resolveStateDir(env, () => "/home/test")).toBe(path.resolve("/new/state"));
  });

  it("falls back to CLAWDBOT_STATE_DIR when MARV_STATE_DIR is unset", () => {
    const env = {
      CLAWDBOT_STATE_DIR: "/legacy/state",
    } as NodeJS.ProcessEnv;

    expect(resolveStateDir(env, () => "/home/test")).toBe(path.resolve("/legacy/state"));
  });

  it("uses MARV_HOME for default state/config locations", () => {
    const env = {
      MARV_HOME: "/srv/marv-home",
    } as NodeJS.ProcessEnv;

    const resolvedHome = path.resolve("/srv/marv-home");
    expect(resolveStateDir(env)).toBe(path.join(resolvedHome, ".marv"));

    const candidates = resolveDefaultConfigCandidates(env);
    expect(candidates[0]).toBe(path.join(resolvedHome, ".marv", "marv.json"));
  });

  it("prefers MARV_HOME over HOME for default state/config locations", () => {
    const env = {
      MARV_HOME: "/srv/marv-home",
      HOME: "/home/other",
    } as NodeJS.ProcessEnv;

    const resolvedHome = path.resolve("/srv/marv-home");
    expect(resolveStateDir(env)).toBe(path.join(resolvedHome, ".marv"));

    const candidates = resolveDefaultConfigCandidates(env);
    expect(candidates[0]).toBe(path.join(resolvedHome, ".marv", "marv.json"));
  });

  it("orders default config candidates in a stable order", () => {
    const home = "/home/test";
    const resolvedHome = path.resolve(home);
    const candidates = resolveDefaultConfigCandidates({} as NodeJS.ProcessEnv, () => home);
    const expected = [
      path.join(resolvedHome, ".marv", "marv.json"),
      path.join(resolvedHome, ".marv", "clawdbot.json"),
      path.join(resolvedHome, ".marv", "moldbot.json"),
      path.join(resolvedHome, ".marv", "moltbot.json"),
      path.join(resolvedHome, ".clawdbot", "marv.json"),
      path.join(resolvedHome, ".clawdbot", "clawdbot.json"),
      path.join(resolvedHome, ".clawdbot", "moldbot.json"),
      path.join(resolvedHome, ".clawdbot", "moltbot.json"),
      path.join(resolvedHome, ".moldbot", "marv.json"),
      path.join(resolvedHome, ".moldbot", "clawdbot.json"),
      path.join(resolvedHome, ".moldbot", "moldbot.json"),
      path.join(resolvedHome, ".moldbot", "moltbot.json"),
      path.join(resolvedHome, ".moltbot", "marv.json"),
      path.join(resolvedHome, ".moltbot", "clawdbot.json"),
      path.join(resolvedHome, ".moltbot", "moldbot.json"),
      path.join(resolvedHome, ".moltbot", "moltbot.json"),
    ];
    expect(candidates).toEqual(expected);
  });

  it("prefers ~/.marv when it exists and legacy dir is missing", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "marv-state-"));
    try {
      const newDir = path.join(root, ".marv");
      await fs.mkdir(newDir, { recursive: true });
      const resolved = resolveStateDir({} as NodeJS.ProcessEnv, () => root);
      expect(resolved).toBe(newDir);
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("CONFIG_PATH prefers existing config when present", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "marv-config-"));
    try {
      const legacyDir = path.join(root, ".marv");
      await fs.mkdir(legacyDir, { recursive: true });
      const legacyPath = path.join(legacyDir, "marv.json");
      await fs.writeFile(legacyPath, "{}", "utf-8");

      const resolved = resolveConfigPathCandidate({} as NodeJS.ProcessEnv, () => root);
      expect(resolved).toBe(legacyPath);
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("respects state dir overrides when config is missing", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "marv-config-override-"));
    try {
      const legacyDir = path.join(root, ".marv");
      await fs.mkdir(legacyDir, { recursive: true });
      const legacyConfig = path.join(legacyDir, "marv.json");
      await fs.writeFile(legacyConfig, "{}", "utf-8");

      const overrideDir = path.join(root, "override");
      const env = { MARV_STATE_DIR: overrideDir } as NodeJS.ProcessEnv;
      const resolved = resolveConfigPath(env, overrideDir, () => root);
      expect(resolved).toBe(path.join(overrideDir, "marv.json"));
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("honors CLAWDBOT_CONFIG_PATH when the canonical config path is unset", () => {
    const env = {
      CLAWDBOT_CONFIG_PATH: "/legacy/marv.json",
    } as NodeJS.ProcessEnv;

    expect(resolveConfigPathCandidate(env, () => "/home/test")).toBe(
      path.resolve("/legacy/marv.json"),
    );
  });
});

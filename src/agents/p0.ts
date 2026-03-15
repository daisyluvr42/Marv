import fs from "node:fs/promises";
import path from "node:path";
import type { MarvConfig } from "../core/config/config.js";
import { resolveAgentWorkspaceDir } from "./agent-scope.js";
import type { EmbeddedContextFile } from "./runner/pi-embedded-helpers.js";

export type AgentP0Section = "soul" | "identity" | "user";

const P0_SOUL_FILENAME = "SOUL.md";
const P0_IDENTITY_FILENAME = "IDENTITY.md";
const P0_USER_FILENAME = "USER.md";

export const P0_SECTION_ORDER: AgentP0Section[] = ["soul", "identity", "user"];

export const P0_FILE_BY_SECTION: Record<AgentP0Section, string> = {
  soul: P0_SOUL_FILENAME,
  identity: P0_IDENTITY_FILENAME,
  user: P0_USER_FILENAME,
};

export const P0_LABEL_BY_SECTION: Record<AgentP0Section, string> = {
  soul: "P0 Soul",
  identity: "P0 Identity",
  user: "P0 User",
};

export function resolveAgentP0Config(cfg: MarvConfig): Partial<Record<AgentP0Section, string>> {
  const p0 = cfg.agents?.defaults?.p0;
  if (!p0 || typeof p0 !== "object") {
    return {};
  }
  return {
    soul: typeof p0.soul === "string" ? p0.soul : undefined,
    identity: typeof p0.identity === "string" ? p0.identity : undefined,
    user: typeof p0.user === "string" ? p0.user : undefined,
  };
}

export function hasConfiguredAgentP0(cfg: MarvConfig): boolean {
  const p0 = resolveAgentP0Config(cfg);
  return P0_SECTION_ORDER.some((section) => typeof p0[section] === "string");
}

export function isP0FileName(name: string): boolean {
  return P0_SECTION_ORDER.some((section) => P0_FILE_BY_SECTION[section] === name);
}

export function resolveP0SectionFromFileName(name: string): AgentP0Section | null {
  return P0_SECTION_ORDER.find((section) => P0_FILE_BY_SECTION[section] === name) ?? null;
}

export function buildP0ContextFiles(cfg: MarvConfig): EmbeddedContextFile[] {
  const p0 = resolveAgentP0Config(cfg);
  return P0_SECTION_ORDER.flatMap((section) => {
    const content = p0[section];
    if (typeof content !== "string" || content.trim().length === 0) {
      return [];
    }
    return [{ path: P0_LABEL_BY_SECTION[section], content }];
  });
}

export function setAgentP0Sections(
  cfg: MarvConfig,
  updates: Partial<Record<AgentP0Section, string | undefined>>,
): MarvConfig {
  const current = resolveAgentP0Config(cfg);
  const nextP0: Partial<Record<AgentP0Section, string>> = {};

  for (const section of P0_SECTION_ORDER) {
    const incoming = updates[section];
    const value = incoming === undefined ? current[section] : incoming;
    if (typeof value === "string") {
      nextP0[section] = value;
    }
  }

  const hasAny = P0_SECTION_ORDER.some((section) => typeof nextP0[section] === "string");
  return {
    ...cfg,
    agents: {
      ...cfg.agents,
      defaults: {
        ...cfg.agents?.defaults,
        p0: hasAny ? nextP0 : undefined,
      },
    },
  };
}

export async function projectAgentP0FilesFromConfig(cfg: MarvConfig): Promise<void> {
  const p0 = resolveAgentP0Config(cfg);
  if (!P0_SECTION_ORDER.some((section) => typeof p0[section] === "string")) {
    return;
  }
  const workspaceDir = resolveAgentWorkspaceDir(cfg, "main");
  await fs.mkdir(workspaceDir, { recursive: true });
  await Promise.all(
    P0_SECTION_ORDER.map(async (section) => {
      const content = p0[section];
      if (typeof content !== "string") {
        return;
      }
      await fs.writeFile(path.join(workspaceDir, P0_FILE_BY_SECTION[section]), content, "utf-8");
    }),
  );
}

export async function readAgentP0FilesFromWorkspace(
  workspaceDir: string,
): Promise<Partial<Record<AgentP0Section, string>>> {
  const next: Partial<Record<AgentP0Section, string>> = {};
  await Promise.all(
    P0_SECTION_ORDER.map(async (section) => {
      try {
        next[section] = await fs.readFile(
          path.join(workspaceDir, P0_FILE_BY_SECTION[section]),
          "utf-8",
        );
      } catch {
        // Missing files are ignored during import.
      }
    }),
  );
  return next;
}

import fs from "node:fs/promises";
import { resolveAgentWorkspaceDir } from "../agents/agent-scope.js";
import {
  P0_FILE_BY_SECTION,
  P0_LABEL_BY_SECTION,
  P0_SECTION_ORDER,
  type AgentP0Section,
  readAgentP0FilesFromWorkspace,
  resolveAgentP0Config,
  setAgentP0Sections,
} from "../agents/p0.js";
import { readConfigFileSnapshotForWrite, writeConfigFile } from "../core/config/config.js";
import { logConfigUpdated } from "../core/config/logging.js";
import type { MarvConfig } from "../core/config/types.js";
import type { RuntimeEnv } from "../runtime.js";
import { defaultRuntime } from "../runtime.js";
import { shortenHomePath } from "../utils.js";

type AgentP0ShowOptions = {
  json?: boolean;
};

type AgentP0SectionOptions = AgentP0ShowOptions & {
  file?: string;
  clear?: boolean;
};

function requireWritableConfig(snapshot: {
  valid: boolean;
  exists: boolean;
  config: MarvConfig;
}): MarvConfig {
  if (snapshot.valid) {
    return snapshot.config;
  }
  if (!snapshot.exists) {
    return {};
  }
  throw new Error("Config is invalid. Run `marv doctor` or `marv configure` first.");
}

function formatP0Snapshot(cfg: MarvConfig) {
  const p0 = resolveAgentP0Config(cfg);
  return {
    soul: p0.soul ?? "",
    identity: p0.identity ?? "",
    user: p0.user ?? "",
  };
}

export async function memoryP0ShowCommand(
  opts: AgentP0ShowOptions = {},
  runtime: RuntimeEnv = defaultRuntime,
) {
  const { snapshot } = await readConfigFileSnapshotForWrite();
  const cfg = requireWritableConfig(snapshot);
  const result = formatP0Snapshot(cfg);

  if (opts.json) {
    runtime.log(JSON.stringify(result, null, 2));
    return;
  }

  for (const section of P0_SECTION_ORDER) {
    runtime.log(`${P0_LABEL_BY_SECTION[section]}:`);
    const content = result[section];
    runtime.log(content.trim().length > 0 ? content : "(empty)");
    if (section !== P0_SECTION_ORDER[P0_SECTION_ORDER.length - 1]) {
      runtime.log("");
    }
  }
}

async function resolveSectionContent(
  value: string | undefined,
  opts: AgentP0SectionOptions,
): Promise<string | undefined> {
  if (opts.clear) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof opts.file === "string" && opts.file.trim()) {
    return await fs.readFile(opts.file, "utf-8");
  }
  return undefined;
}

export async function memoryP0SectionCommand(
  section: AgentP0Section,
  value: string | undefined,
  opts: AgentP0SectionOptions = {},
  runtime: RuntimeEnv = defaultRuntime,
) {
  const { snapshot, writeOptions } = await readConfigFileSnapshotForWrite();
  const cfg = requireWritableConfig(snapshot);
  const nextValue = await resolveSectionContent(value, opts);

  if (nextValue === undefined) {
    const current = resolveAgentP0Config(cfg)[section] ?? "";
    if (opts.json) {
      runtime.log(
        JSON.stringify(
          {
            section,
            label: P0_LABEL_BY_SECTION[section],
            file: P0_FILE_BY_SECTION[section],
            value: current,
          },
          null,
          2,
        ),
      );
      return;
    }
    runtime.log(`${P0_LABEL_BY_SECTION[section]}:`);
    runtime.log(current.trim().length > 0 ? current : "(empty)");
    return;
  }

  const nextConfig = setAgentP0Sections(cfg, { [section]: nextValue });
  await writeConfigFile(nextConfig, writeOptions);

  if (opts.json) {
    runtime.log(
      JSON.stringify(
        {
          ok: true,
          section,
          file: P0_FILE_BY_SECTION[section],
          value: nextValue,
        },
        null,
        2,
      ),
    );
    return;
  }

  logConfigUpdated(runtime);
  runtime.log(`${P0_LABEL_BY_SECTION[section]} updated.`);
  runtime.log(`Projection: ${P0_FILE_BY_SECTION[section]}`);
}

export async function memoryP0SyncCommand(
  opts: AgentP0ShowOptions = {},
  runtime: RuntimeEnv = defaultRuntime,
) {
  const { snapshot, writeOptions } = await readConfigFileSnapshotForWrite();
  const cfg = requireWritableConfig(snapshot);
  const workspaceDir = resolveAgentWorkspaceDir(cfg, "main");
  const imported = await readAgentP0FilesFromWorkspace(workspaceDir);
  const hasAny = P0_SECTION_ORDER.some((section) => typeof imported[section] === "string");

  if (!hasAny) {
    throw new Error(`No P0 files found in ${shortenHomePath(workspaceDir)}.`);
  }

  const nextConfig = setAgentP0Sections(cfg, imported);
  await writeConfigFile(nextConfig, writeOptions);

  if (opts.json) {
    runtime.log(
      JSON.stringify(
        {
          ok: true,
          workspace: workspaceDir,
          imported: formatP0Snapshot(nextConfig),
        },
        null,
        2,
      ),
    );
    return;
  }

  logConfigUpdated(runtime);
  runtime.log(`Synced P0 from ${shortenHomePath(workspaceDir)}.`);
}

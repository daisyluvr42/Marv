import fs from "node:fs/promises";
import path from "node:path";
import { resolveDefaultAgentId } from "../agents/agent-scope.js";
import { identityHasValues, parseIdentityMarkdown } from "../agents/prompt/identity-file.js";
import { DEFAULT_IDENTITY_FILENAME } from "../agents/workspace.js";
import { writeConfigFile } from "../core/config/config.js";
import { logConfigUpdated } from "../core/config/logging.js";
import type { IdentityConfig } from "../core/config/types.js";
import { normalizeAgentId } from "../routing/session-key.js";
import type { RuntimeEnv } from "../runtime.js";
import { defaultRuntime } from "../runtime.js";
import { resolveUserPath, shortenHomePath } from "../utils.js";
import { requireValidConfig } from "./agents.command-shared.js";
import { type AgentIdentity, loadAgentIdentity } from "./agents.config.js";

type AgentsSetIdentityOptions = {
  agent?: string;
  workspace?: string;
  identityFile?: string;
  name?: string;
  emoji?: string;
  theme?: string;
  avatar?: string;
  fromIdentity?: boolean;
  json?: boolean;
};

const normalizeWorkspacePath = (input: string) => path.resolve(resolveUserPath(input));

const coerceTrimmed = (value?: string) => {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
};

async function loadIdentityFromFile(filePath: string): Promise<AgentIdentity | null> {
  try {
    const content = await fs.readFile(filePath, "utf-8");
    const parsed = parseIdentityMarkdown(content);
    if (!identityHasValues(parsed)) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export async function agentsSetIdentityCommand(
  opts: AgentsSetIdentityOptions,
  runtime: RuntimeEnv = defaultRuntime,
) {
  const cfg = await requireValidConfig(runtime);
  if (!cfg) {
    return;
  }

  const agentRaw = coerceTrimmed(opts.agent);
  const nameRaw = coerceTrimmed(opts.name);
  const emojiRaw = coerceTrimmed(opts.emoji);
  const themeRaw = coerceTrimmed(opts.theme);
  const avatarRaw = coerceTrimmed(opts.avatar);
  const hasExplicitIdentity = Boolean(nameRaw || emojiRaw || themeRaw || avatarRaw);

  const identityFileRaw = coerceTrimmed(opts.identityFile);
  const workspaceRaw = coerceTrimmed(opts.workspace);
  const wantsIdentityFile = Boolean(opts.fromIdentity || identityFileRaw || !hasExplicitIdentity);

  let identityFilePath: string | undefined;
  let workspaceDir: string | undefined;

  if (identityFileRaw) {
    identityFilePath = normalizeWorkspacePath(identityFileRaw);
    workspaceDir = path.dirname(identityFilePath);
  } else if (workspaceRaw) {
    workspaceDir = normalizeWorkspacePath(workspaceRaw);
  } else if (wantsIdentityFile || !agentRaw) {
    workspaceDir = path.resolve(process.cwd());
  }

  const agentId = agentRaw
    ? normalizeAgentId(agentRaw)
    : normalizeAgentId(resolveDefaultAgentId(cfg));
  if (agentId !== normalizeAgentId(resolveDefaultAgentId(cfg))) {
    runtime.error(
      'Only the durable "main" agent can be configured. Use enhanced subagents for delegated roles.',
    );
    runtime.exit(1);
    return;
  }

  let identityFromFile: AgentIdentity | null = null;
  if (wantsIdentityFile) {
    if (identityFilePath) {
      identityFromFile = await loadIdentityFromFile(identityFilePath);
    } else if (workspaceDir) {
      identityFromFile = loadAgentIdentity(workspaceDir);
    }
    if (!identityFromFile) {
      const targetPath =
        identityFilePath ??
        (workspaceDir ? path.join(workspaceDir, DEFAULT_IDENTITY_FILENAME) : "IDENTITY.md");
      runtime.error(`No identity data found in ${shortenHomePath(targetPath)}.`);
      runtime.exit(1);
      return;
    }
  }

  const fileTheme =
    identityFromFile?.theme ?? identityFromFile?.creature ?? identityFromFile?.vibe ?? undefined;
  const incomingIdentity: IdentityConfig = {
    ...(nameRaw || identityFromFile?.name ? { name: nameRaw ?? identityFromFile?.name } : {}),
    ...(emojiRaw || identityFromFile?.emoji ? { emoji: emojiRaw ?? identityFromFile?.emoji } : {}),
    ...(themeRaw || fileTheme ? { theme: themeRaw ?? fileTheme } : {}),
    ...(avatarRaw || identityFromFile?.avatar
      ? { avatar: avatarRaw ?? identityFromFile?.avatar }
      : {}),
  };

  if (
    !incomingIdentity.name &&
    !incomingIdentity.emoji &&
    !incomingIdentity.theme &&
    !incomingIdentity.avatar
  ) {
    runtime.error(
      "No identity fields provided. Use --name/--emoji/--theme/--avatar or --from-identity.",
    );
    runtime.exit(1);
    return;
  }

  const nextIdentity: IdentityConfig = {
    ...cfg.agents?.defaults?.identity,
    ...incomingIdentity,
  };

  const nextConfig = {
    ...cfg,
    agents: {
      ...cfg.agents,
      defaults: {
        ...cfg.agents?.defaults,
        identity: nextIdentity,
      },
    },
  };

  await writeConfigFile(nextConfig);

  if (opts.json) {
    runtime.log(
      JSON.stringify(
        {
          agentId,
          identity: nextIdentity,
          workspace: workspaceDir ?? null,
          identityFile: identityFilePath ?? null,
        },
        null,
        2,
      ),
    );
    return;
  }

  logConfigUpdated(runtime);
  runtime.log(`Agent: ${agentId}`);
  if (nextIdentity.name) {
    runtime.log(`Name: ${nextIdentity.name}`);
  }
  if (nextIdentity.theme) {
    runtime.log(`Theme: ${nextIdentity.theme}`);
  }
  if (nextIdentity.emoji) {
    runtime.log(`Emoji: ${nextIdentity.emoji}`);
  }
  if (nextIdentity.avatar) {
    runtime.log(`Avatar: ${nextIdentity.avatar}`);
  }
  if (workspaceDir) {
    runtime.log(`Workspace: ${shortenHomePath(workspaceDir)}`);
  }
}

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { resolveCliName } from "../cli/cli-name.js";
import { resolveMarvPackageRoot } from "../infra/marv-root.js";
import type { RuntimeEnv } from "../runtime.js";
import { note } from "../terminal/note.js";
import type { DoctorPrompter } from "./doctor-prompter.js";

// Late-bound import to break circular dependency chain
// (completion-cli -> command-registry -> register.maintenance -> doctor -> doctor-completion -> completion-cli).
type CompletionModule = {
  resolveShellFromEnv: (env?: NodeJS.ProcessEnv) => string;
  resolveCompletionCachePath: (shell: string, binName: string) => string;
  completionCacheExists: (shell: string, binName: string) => Promise<boolean>;
  isCompletionInstalled: (shell: string, binName: string) => Promise<boolean>;
  usesSlowDynamicCompletion: (shell: string, binName: string) => Promise<boolean>;
  installCompletion: (shell: string, yes: boolean, binName?: string) => Promise<void>;
};
const COMPLETION_MODULE = "../cli/completion-cli.js";
async function loadCompletionModule(): Promise<CompletionModule> {
  return import(/* @vite-ignore */ COMPLETION_MODULE);
}

type CompletionShell = "zsh" | "bash" | "fish" | "powershell";

/** Generate the completion cache by spawning the CLI. */
async function generateCompletionCache(): Promise<boolean> {
  const root = await resolveMarvPackageRoot({
    moduleUrl: import.meta.url,
    argv1: process.argv[1],
    cwd: process.cwd(),
  });
  if (!root) {
    return false;
  }

  const binPath = [path.join(root, "marv.mjs"), path.join(root, "marv.mjs")].find((candidate) =>
    fs.existsSync(candidate),
  );
  if (!binPath) {
    return false;
  }
  const result = spawnSync(process.execPath, [binPath, "completion", "--write-state"], {
    cwd: root,
    env: process.env,
    encoding: "utf-8",
  });

  return result.status === 0;
}

export type ShellCompletionStatus = {
  shell: CompletionShell;
  profileInstalled: boolean;
  cacheExists: boolean;
  cachePath: string;
  /** True if profile uses slow dynamic pattern like `source <(marv completion ...)` */
  usesSlowPattern: boolean;
};

/** Check the status of shell completion for the current shell. */
export async function checkShellCompletionStatus(binName = "marv"): Promise<ShellCompletionStatus> {
  const mod = await loadCompletionModule();
  const shell = mod.resolveShellFromEnv() as CompletionShell;
  const profileInstalled = await mod.isCompletionInstalled(shell, binName);
  const cacheExists = await mod.completionCacheExists(shell, binName);
  const cachePath = mod.resolveCompletionCachePath(shell, binName);
  const usesSlowPattern = await mod.usesSlowDynamicCompletion(shell, binName);

  return {
    shell,
    profileInstalled,
    cacheExists,
    cachePath,
    usesSlowPattern,
  };
}

export type DoctorCompletionOptions = {
  nonInteractive?: boolean;
};

/**
 * Doctor check for shell completion.
 * - If profile uses slow dynamic pattern: upgrade to cached version
 * - If profile has completion but no cache: auto-generate cache and upgrade profile
 * - If no completion at all: prompt to install (with user confirmation)
 */
export async function doctorShellCompletion(
  runtime: RuntimeEnv,
  prompter: DoctorPrompter,
  options: DoctorCompletionOptions = {},
): Promise<void> {
  const mod = await loadCompletionModule();
  const cliName = resolveCliName();
  const status = await checkShellCompletionStatus(cliName);

  // Profile uses slow dynamic pattern - upgrade to cached version
  if (status.usesSlowPattern) {
    note(
      `Your ${status.shell} profile uses slow dynamic completion (source <(...)).\nUpgrading to cached completion for faster shell startup...`,
      "Shell completion",
    );

    // Ensure cache exists first
    if (!status.cacheExists) {
      const generated = await generateCompletionCache();
      if (!generated) {
        note(
          `Failed to generate completion cache. Run \`${cliName} completion --write-state\` manually.`,
          "Shell completion",
        );
        return;
      }
    }

    // Upgrade profile to use cached file
    await mod.installCompletion(status.shell, true, cliName);
    note(
      `Shell completion upgraded. Restart your shell or run: source ~/.${status.shell === "zsh" ? "zshrc" : status.shell === "bash" ? "bashrc" : "config/fish/config.fish"}`,
      "Shell completion",
    );
    return;
  }

  // Profile has completion but no cache - auto-fix
  if (status.profileInstalled && !status.cacheExists) {
    note(
      `Shell completion is configured in your ${status.shell} profile but the cache is missing.\nRegenerating cache...`,
      "Shell completion",
    );
    const generated = await generateCompletionCache();
    if (generated) {
      note(`Completion cache regenerated at ${status.cachePath}`, "Shell completion");
    } else {
      note(
        `Failed to regenerate completion cache. Run \`${cliName} completion --write-state\` manually.`,
        "Shell completion",
      );
    }
    return;
  }

  // No completion at all - prompt to install
  if (!status.profileInstalled) {
    if (options.nonInteractive) {
      // In non-interactive mode, just note that completion is not installed
      return;
    }

    const shouldInstall = await prompter.confirm({
      message: `Enable ${status.shell} shell completion for ${cliName}?`,
      initialValue: true,
    });

    if (shouldInstall) {
      // First generate the cache
      const generated = await generateCompletionCache();
      if (!generated) {
        note(
          `Failed to generate completion cache. Run \`${cliName} completion --write-state\` manually.`,
          "Shell completion",
        );
        return;
      }

      // Then install to profile
      await mod.installCompletion(status.shell, true, cliName);
      note(
        `Shell completion installed. Restart your shell or run: source ~/.${status.shell === "zsh" ? "zshrc" : status.shell === "bash" ? "bashrc" : "config/fish/config.fish"}`,
        "Shell completion",
      );
    }
  }
}

/**
 * Ensure completion cache exists. Used during onboarding/update to fix
 * cases where profile has completion but no cache.
 * This is a silent fix - no prompts.
 */
export async function ensureCompletionCacheExists(binName = "marv"): Promise<boolean> {
  const mod = await loadCompletionModule();
  const resolvedBinName = binName === "marv" ? "marv" : binName;
  const shell = mod.resolveShellFromEnv() as CompletionShell;
  const cacheExists = await mod.completionCacheExists(shell, resolvedBinName);

  if (cacheExists) {
    return true;
  }

  return generateCompletionCache();
}

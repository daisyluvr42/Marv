import { formatCliCommand } from "../cli/command-format.js";
import { readConfigFileSnapshot } from "../core/config/config.js";
import { assertSupportedRuntime } from "../infra/runtime-guard.js";
import type { RuntimeEnv } from "../runtime.js";
import { defaultRuntime } from "../runtime.js";
import { resolveUserPath } from "../utils.js";
import { DEFAULT_WORKSPACE, handleReset } from "./onboard-helpers.js";
import { runInteractiveOnboarding } from "./onboard-interactive.js";
import { runNonInteractiveOnboarding } from "./onboard-non-interactive.js";
import type { OnboardOptions } from "./onboard-types.js";

export async function onboardCommand(opts: OnboardOptions, runtime: RuntimeEnv = defaultRuntime) {
  assertSupportedRuntime(runtime);
  if (opts.nonInteractive && opts.acceptRisk !== true) {
    runtime.error(
      [
        "Non-interactive onboarding requires explicit risk acknowledgement.",
        "Read: /security",
        `Re-run with: ${formatCliCommand("marv onboard --non-interactive --accept-risk ...")}`,
      ].join("\n"),
    );
    runtime.exit(1);
    return;
  }

  if (opts.reset) {
    const snapshot = await readConfigFileSnapshot();
    const baseConfig = snapshot.valid ? snapshot.config : {};
    const workspaceDefault =
      opts.workspace ?? baseConfig.agents?.defaults?.workspace ?? DEFAULT_WORKSPACE;
    await handleReset("full", resolveUserPath(workspaceDefault), runtime);
  }

  if (process.platform === "win32") {
    runtime.log(
      [
        "Windows detected — Marv runs great on WSL2!",
        "Native Windows might be trickier.",
        "Quick setup: wsl --install (one command, one reboot)",
        "Guide: /windows",
      ].join("\n"),
    );
  }

  if (opts.nonInteractive) {
    await runNonInteractiveOnboarding(opts, runtime);
    return;
  }

  await runInteractiveOnboarding(opts, runtime);
}

export type { OnboardOptions } from "./onboard-types.js";

import { resolveGatewayService } from "../../infra/daemon/service.js";
import { runGatewayRollback } from "../../infra/update/update-runner.js";
import { defaultRuntime } from "../../runtime.js";
import { theme } from "../../terminal/theme.js";
import { createUpdateProgress, printResult } from "./progress.js";
import { prepareRestartScript } from "./restart-helper.js";
import { parseTimeoutMsOrExit, resolveUpdateRoot, type UpdateRollbackOptions } from "./shared.js";
import { suppressDeprecations } from "./suppress-deprecations.js";
import { maybeRestartService } from "./update-command.js";

export async function updateRollbackCommand(opts: UpdateRollbackOptions): Promise<void> {
  suppressDeprecations();

  const timeoutMs = parseTimeoutMsOrExit(opts.timeout);
  const shouldRestart = opts.restart !== false;
  if (timeoutMs === null) {
    return;
  }

  const root = await resolveUpdateRoot();

  const showProgress = !opts.json && process.stdout.isTTY;
  if (!opts.json) {
    defaultRuntime.log(theme.heading("Rolling back Marv..."));
    defaultRuntime.log("");
  }

  const { progress, stop } = createUpdateProgress(showProgress);
  let restartScriptPath: string | null = null;
  if (shouldRestart) {
    try {
      const loaded = await resolveGatewayService().isLoaded({ env: process.env });
      if (loaded) {
        restartScriptPath = await prepareRestartScript(process.env);
      }
    } catch {
      // Ignore service probe failures; fallback to the standard restart path.
    }
  }

  const result = await runGatewayRollback({
    cwd: root,
    argv1: process.argv[1],
    timeoutMs: timeoutMs ?? undefined,
    progress,
  });

  stop();
  printResult(result, { ...opts, hideSteps: showProgress });

  if (result.status === "error") {
    defaultRuntime.exit(1);
    return;
  }

  if (result.status === "skipped") {
    if (!opts.json) {
      if (result.reason === "no-last-known-good") {
        defaultRuntime.log(
          theme.warn(
            "Skipped: no last-known-good deployment is recorded yet, so there is nothing safe to roll back to.",
          ),
        );
      } else if (result.reason === "not-git-install") {
        defaultRuntime.log(
          theme.warn(
            "Skipped: rollback rescue only works for git deployments with recorded deploy state.",
          ),
        );
      }
    }
    defaultRuntime.exit(0);
    return;
  }

  await maybeRestartService({
    shouldRestart,
    result,
    opts,
    restartScriptPath,
  });

  if (!opts.json) {
    defaultRuntime.log(theme.muted("Rolled back to the last known good deployment."));
  }
}

import type { RuntimeEnv } from "../runtime.js";
import { defaultRuntime } from "../runtime.js";
import { requireValidConfig } from "./agents.command-shared.js";

type AgentsAddOptions = {
  name?: string;
  workspace?: string;
  model?: string;
  agentDir?: string;
  bind?: string[];
  nonInteractive?: boolean;
  json?: boolean;
};

export async function agentsAddCommand(
  opts: AgentsAddOptions,
  runtime: RuntimeEnv = defaultRuntime,
  params?: { hasFlags?: boolean },
) {
  void opts;
  void params;
  if (!(await requireValidConfig(runtime))) {
    return;
  }
  runtime.error(
    'Top-level "marv agents add" is no longer supported. Configure "main" under agents.defaults and use enhanced subagents for delegation.',
  );
  runtime.exit(1);
}

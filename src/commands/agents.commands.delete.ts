import type { RuntimeEnv } from "../runtime.js";
import { defaultRuntime } from "../runtime.js";
import { requireValidConfig } from "./agents.command-shared.js";

type AgentsDeleteOptions = {
  id: string;
  force?: boolean;
  json?: boolean;
};

export async function agentsDeleteCommand(
  opts: AgentsDeleteOptions,
  runtime: RuntimeEnv = defaultRuntime,
) {
  void opts;
  if (!(await requireValidConfig(runtime))) {
    return;
  }
  runtime.error(
    'Top-level "marv agents delete" is no longer supported. Marv now keeps a single durable "main" agent.',
  );
  runtime.exit(1);
}

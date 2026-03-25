import type { MarvConfig } from "../../../core/config/config.js";
import type { TurnContext } from "../../support/templating.js";
import { buildCommandTestParams as buildBaseCommandTestParams } from "../dispatch-harness.js";

export function buildCommandTestParams(
  commandBody: string,
  cfg: MarvConfig,
  ctxOverrides?: Partial<TurnContext>,
) {
  return buildBaseCommandTestParams(commandBody, cfg, ctxOverrides);
}

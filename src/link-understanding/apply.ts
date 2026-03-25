import { finalizeInboundContext } from "../auto-reply/inbound/context.js";
import type { TurnContext } from "../auto-reply/support/templating.js";
import type { MarvConfig } from "../core/config/config.js";
import { formatLinkUnderstandingBody } from "./format.js";
import { runLinkUnderstanding } from "./runner.js";

export type ApplyLinkUnderstandingResult = {
  outputs: string[];
  urls: string[];
};

export async function applyLinkUnderstanding(params: {
  ctx: TurnContext;
  cfg: MarvConfig;
}): Promise<ApplyLinkUnderstandingResult> {
  const result = await runLinkUnderstanding({
    cfg: params.cfg,
    ctx: params.ctx,
  });

  if (result.outputs.length === 0) {
    return result;
  }

  params.ctx.LinkUnderstanding = [...(params.ctx.LinkUnderstanding ?? []), ...result.outputs];
  params.ctx.Body = formatLinkUnderstandingBody({
    body: params.ctx.Body,
    outputs: result.outputs,
  });

  finalizeInboundContext(params.ctx, {
    forceBodyForAgent: true,
    forceBodyForCommands: true,
  });

  return result;
}

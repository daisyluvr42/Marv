import type { MarvConfig } from "../../core/config/config.js";
import { applyLinkUnderstanding } from "../../link-understanding/apply.js";
import { applyMediaUnderstanding } from "../../media-understanding/apply.js";
import { resolveCommandAuthorization } from "../commands/auth.js";
import type { FinalizedTurnContext, TurnContext } from "../support/templating.js";
import { finalizeInboundContext } from "./context.js";

export type EnrichResult = {
  finalized: FinalizedTurnContext;
  commandAuthorized: boolean;
};

/**
 * Stage 2: Finalize the inbound context, run media/link understanding,
 * and resolve command authorization.
 */
export async function enrichInbound(params: {
  ctx: TurnContext;
  cfg: MarvConfig;
  agentDir: string;
  provider: string;
  model: string;
  isFastTestEnv: boolean;
}): Promise<EnrichResult> {
  const { ctx, cfg, agentDir, provider, model, isFastTestEnv } = params;

  const finalized = finalizeInboundContext(ctx);

  if (!isFastTestEnv) {
    await Promise.all([
      applyMediaUnderstanding({
        ctx: finalized,
        cfg,
        agentDir,
        activeModel: { provider, model },
      }),
      applyLinkUnderstanding({
        ctx: finalized,
        cfg,
      }),
    ]);
  }

  const commandAuthorized = finalized.CommandAuthorized;
  resolveCommandAuthorization({
    ctx: finalized,
    cfg,
    commandAuthorized,
  });

  return { finalized, commandAuthorized };
}

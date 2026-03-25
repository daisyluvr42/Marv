import type { MarvConfig } from "../../core/config/config.js";
import { loadPromptMediaImages } from "../support/prompt-media-images.js";
import { stageSandboxMedia } from "../support/stage-sandbox-media.js";
import type { TurnContext, SessionTemplateContext } from "../support/templating.js";
import type { GetReplyOptions } from "../support/types.js";

export type PrepareExecutionResult = {
  runReplyOpts: GetReplyOptions | undefined;
};

/**
 * Stage 6: Stage sandbox media and load prompt media images
 * before handing off to the agent runner.
 */
export async function prepareExecution(params: {
  ctx: TurnContext;
  sessionCtx: SessionTemplateContext;
  cfg: MarvConfig;
  sessionKey: string;
  workspaceDir: string;
  resolvedOpts: GetReplyOptions | undefined;
}): Promise<PrepareExecutionResult> {
  const { ctx, sessionCtx, cfg, sessionKey, workspaceDir, resolvedOpts } = params;

  await stageSandboxMedia({
    ctx,
    sessionCtx,
    cfg,
    sessionKey,
    workspaceDir,
  });

  const inboundImages = await loadPromptMediaImages({
    ctx,
    workspaceDir,
    existingImages: resolvedOpts?.images,
  });
  const runReplyOpts =
    inboundImages !== resolvedOpts?.images
      ? { ...resolvedOpts, images: inboundImages }
      : resolvedOpts;

  return { runReplyOpts };
}

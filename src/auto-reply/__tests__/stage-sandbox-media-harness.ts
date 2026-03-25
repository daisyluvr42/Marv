import { join } from "node:path";
import { withTempHome as withTempHomeBase } from "../../../test/helpers/temp-home.js";
import type { MarvConfig } from "../../core/config/config.js";
import type { TurnContext, SessionTemplateContext } from "../support/templating.js";

export async function withSandboxMediaTempHome<T>(
  prefix: string,
  fn: (home: string) => Promise<T>,
): Promise<T> {
  return withTempHomeBase(async (home) => await fn(home), { prefix });
}

export function createSandboxMediaContexts(mediaPath: string): {
  ctx: TurnContext;
  sessionCtx: SessionTemplateContext;
} {
  const ctx: TurnContext = {
    Body: "hi",
    From: "whatsapp:group:demo",
    To: "+2000",
    ChatType: "group",
    Provider: "whatsapp",
    MediaPath: mediaPath,
    MediaType: "image/jpeg",
    MediaUrl: mediaPath,
  };
  return { ctx, sessionCtx: { ...ctx } };
}

export function createSandboxMediaStageConfig(home: string): MarvConfig {
  return {
    agents: {
      defaults: {
        model: "anthropic/claude-opus-4-5",
        workspace: join(home, "marv"),
        sandbox: {
          mode: "non-main",
          workspaceRoot: join(home, "sandboxes"),
        },
      },
    },
    channels: { whatsapp: { allowFrom: ["*"] } },
    session: { store: join(home, "sessions.json") },
  } as MarvConfig;
}

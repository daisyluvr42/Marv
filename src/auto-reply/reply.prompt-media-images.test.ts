import fs from "node:fs/promises";
import path from "node:path";
import "./reply.directive.directive-behavior.e2e-mocks.js";
import { describe, expect, it, vi } from "vitest";
import { withTempHome as withTempHomeBase } from "../../test/helpers/temp-home.js";
import { runEmbeddedPiAgent } from "../agents/runner/pi-embedded.js";
import type { MarvConfig } from "../core/config/config.js";
import { getReplyFromConfig } from "./reply.js";

const ONE_PIXEL_PNG_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aR1EAAAAASUVORK5CYII=";

function makeResult(text: string) {
  return {
    payloads: [{ text }],
    meta: {
      durationMs: 5,
      agentMeta: { sessionId: "s", provider: "p", model: "m" },
    },
  };
}

async function withTempHome<T>(fn: (home: string) => Promise<T>): Promise<T> {
  return withTempHomeBase(
    async (home) => {
      vi.mocked(runEmbeddedPiAgent).mockReset();
      return await fn(home);
    },
    {
      env: {
        MARV_BUNDLED_SKILLS_DIR: (home) => path.join(home, "bundled-skills"),
        MARV_TEST_FAST: () => "1",
      },
      prefix: "marv-prompt-media-images-",
    },
  );
}

function makeCfg(home: string) {
  return {
    agents: {
      defaults: {
        model: "anthropic/claude-opus-4-5",
        workspace: path.join(home, "marv"),
      },
    },
    channels: { whatsapp: { allowFrom: ["*"] } },
    session: { store: path.join(home, "sessions.json") },
  } as unknown as MarvConfig;
}

describe("getReplyFromConfig prompt media images", () => {
  it("passes multimodal prompt images through to the embedded runner", async () => {
    await withTempHome(async (home) => {
      const imagePath = path.join(home, "attached.png");
      await fs.writeFile(imagePath, Buffer.from(ONE_PIXEL_PNG_B64, "base64"));

      let seenImages:
        | Array<{
            type: "image";
            data: string;
            mimeType: string;
          }>
        | undefined;
      vi.mocked(runEmbeddedPiAgent).mockImplementation(async (params) => {
        seenImages = params.images;
        return makeResult("ok");
      });

      const cfg = makeCfg(home);
      const res = await getReplyFromConfig(
        {
          Body: "what is in this image?",
          From: "+1001",
          To: "+2000",
          MultimodalRouting: {
            promptMedia: [{ kind: "image", source: "native", path: imagePath }],
            derivedText: {},
            decisions: [],
            settled: true,
          },
        },
        {},
        cfg,
      );

      const text = Array.isArray(res) ? res[0]?.text : res?.text;
      expect(text).toBe("ok");
      expect(seenImages).toHaveLength(1);
      expect(seenImages?.[0]).toMatchObject({
        type: "image",
        mimeType: "image/png",
      });
      expect(seenImages?.[0]?.data).toBeTruthy();
    });
  });

  it("treats prompt-media-only messages as valid inbound media turns", async () => {
    await withTempHome(async (home) => {
      const imagePath = path.join(home, "attached-empty-body.png");
      await fs.writeFile(imagePath, Buffer.from(ONE_PIXEL_PNG_B64, "base64"));

      let seenPrompt: string | undefined;
      vi.mocked(runEmbeddedPiAgent).mockImplementation(async (params) => {
        seenPrompt = params.prompt;
        return makeResult("ok");
      });

      const cfg = makeCfg(home);
      const res = await getReplyFromConfig(
        {
          Body: "",
          From: "+1001",
          To: "+2000",
          MultimodalRouting: {
            promptMedia: [{ kind: "image", source: "native", path: imagePath }],
            derivedText: {},
            decisions: [],
            settled: true,
          },
        },
        {},
        cfg,
      );

      const text = Array.isArray(res) ? res[0]?.text : res?.text;
      expect(text).toBe("ok");
      expect(seenPrompt).toContain("[User sent media without caption]");
    });
  });
});

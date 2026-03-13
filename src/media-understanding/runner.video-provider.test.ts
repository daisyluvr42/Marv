import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { MsgContext } from "../auto-reply/templating.js";
import type { MarvConfig } from "../core/config/config.js";
import {
  buildProviderRegistry,
  createMediaAttachmentCache,
  normalizeMediaAttachments,
  runCapability,
} from "./runner.js";

describe("runCapability video provider options", () => {
  it("merges provider options, headers, and baseUrl overrides", async () => {
    const tmpPath = path.join(os.tmpdir(), `marv-video-${Date.now()}.mp4`);
    await fs.writeFile(tmpPath, Buffer.from("video"));
    const ctx: MsgContext = { MediaPath: tmpPath, MediaType: "video/mp4" };
    const media = normalizeMediaAttachments(ctx);
    const cache = createMediaAttachmentCache(media);

    let seenBaseUrl: string | undefined;
    let seenHeaders: Record<string, string> | undefined;

    const providerRegistry = buildProviderRegistry({
      google: {
        id: "google",
        capabilities: ["video"],
        describeVideo: async (req) => {
          seenBaseUrl = req.baseUrl;
          seenHeaders = req.headers;
          return { text: "video ok", model: req.model };
        },
      },
    });

    const cfg = {
      models: {
        providers: {
          google: {
            baseUrl: "https://provider.example",
            apiKey: "test-key",
            headers: { "X-Provider": "1" },
            models: [],
          },
        },
      },
      tools: {
        media: {
          video: {
            enabled: true,
            baseUrl: "https://config.example",
            headers: { "X-Config": "2" },
            models: [
              {
                provider: "google",
                model: "gemini-video",
                baseUrl: "https://entry.example",
                headers: { "X-Entry": "3" },
              },
            ],
          },
        },
      },
    } as unknown as MarvConfig;

    try {
      const result = await runCapability({
        capability: "video",
        cfg,
        ctx,
        attachments: cache,
        media,
        providerRegistry,
      });
      expect(result.outputs[0]?.text).toBe("video ok");
      expect(seenBaseUrl).toBe("https://entry.example");
      expect(seenHeaders).toMatchObject({
        "X-Provider": "1",
        "X-Config": "2",
        "X-Entry": "3",
      });
    } finally {
      await cache.cleanup();
      await fs.unlink(tmpPath).catch(() => {});
    }
  });
});

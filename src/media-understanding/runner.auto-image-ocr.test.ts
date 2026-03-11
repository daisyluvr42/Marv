import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MsgContext } from "../auto-reply/templating.js";
import type { MarvConfig } from "../core/config/config.js";
import { runExec } from "../process/exec.js";
import {
  buildProviderRegistry,
  clearMediaUnderstandingBinaryCacheForTests,
  createMediaAttachmentCache,
  normalizeMediaAttachments,
  runCapability,
} from "./runner.js";

vi.mock("../process/exec.js", () => ({
  runExec: vi.fn(),
}));

const maybeIt = process.platform === "darwin" ? it : it.skip;

describe("runCapability auto image OCR", () => {
  const mockedRunExec = vi.mocked(runExec);
  let originalPath = "";

  beforeEach(() => {
    clearMediaUnderstandingBinaryCacheForTests();
    mockedRunExec.mockReset();
    originalPath = process.env.PATH ?? "";
  });

  afterEach(() => {
    process.env.PATH = originalPath;
    clearMediaUnderstandingBinaryCacheForTests();
  });

  maybeIt("uses local OCR when no image model is configured", async () => {
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-ocr-bin-"));
    const imagePath = path.join(tempDir, "screen.png");
    const xcrunPath = path.join(tempDir, "xcrun");
    await fs.writeFile(xcrunPath, "#!/bin/sh\nexit 0\n", { mode: 0o755 });
    await fs.writeFile(imagePath, "fake-image");
    process.env.PATH = tempDir;
    mockedRunExec.mockResolvedValue({
      stdout: "Meeting notes\nShip dashboard first",
      stderr: "",
      exitCode: 0,
      signal: null,
      durationMs: 10,
    });
    const ctx: MsgContext = { MediaPath: imagePath, MediaType: "image/png" };
    const media = normalizeMediaAttachments(ctx);
    const cache = createMediaAttachmentCache(media);
    const cfg = {} as MarvConfig;

    try {
      const result = await runCapability({
        capability: "image",
        cfg,
        ctx,
        attachments: cache,
        media,
        providerRegistry: buildProviderRegistry(),
      });

      expect(result.outputs).toHaveLength(1);
      expect(result.outputs[0]?.text).toBe("Meeting notes\nShip dashboard first");
      expect(result.outputs[0]?.provider).toBe("cli");
      expect(result.decision.outcome).toBe("success");
      expect(mockedRunExec).toHaveBeenCalledWith(
        "xcrun",
        expect.arrayContaining(["swift", expect.stringContaining("marv-ocr.swift"), imagePath]),
        expect.any(Object),
      );
    } finally {
      await cache.cleanup();
      await fs.rm(tempDir, { recursive: true, force: true });
    }
  });
});

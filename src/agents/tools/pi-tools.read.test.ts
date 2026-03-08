import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Type } from "@sinclair/typebox";
import { describe, expect, it, vi } from "vitest";
import { createMarvReadTool } from "./pi-tools.read.js";

describe("createMarvReadTool", () => {
  it("returns MIME guidance when read output looks like binary garbage", async () => {
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-read-binary-"));
    const zipPath = path.join(tmpDir, "archive.zip");
    await fs.writeFile(zipPath, Buffer.from([0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00]));
    const baseRead: AgentTool = {
      name: "read",
      label: "read",
      description: "test read",
      parameters: Type.Object({
        path: Type.String(),
      }),
      execute: vi.fn(async () => ({
        content: [{ type: "text" as const, text: "\uFFFD\uFFFD\uFFFD\u0000" }],
        details: { path: zipPath },
      })),
    };

    try {
      const wrapped = createMarvReadTool(
        baseRead as unknown as Parameters<typeof createMarvReadTool>[0],
      );
      const result = await wrapped.execute("read-binary-1", { path: zipPath });
      const details = result.details as {
        ok: boolean;
        detectedMimeType?: string;
        detectedExtension?: string;
        synthesisHint?: string;
      };

      expect(details.ok).toBe(false);
      expect(details.detectedMimeType).toBe("application/zip");
      expect(details.detectedExtension).toBe(".zip");
      expect(details.synthesisHint).toContain("unzip");
    } finally {
      await fs.rm(tmpDir, { recursive: true, force: true });
    }
  });
});

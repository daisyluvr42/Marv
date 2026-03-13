import crypto from "node:crypto";
import fs from "node:fs/promises";
import type { Server } from "node:http";
import path from "node:path";
import express, { type Express } from "express";
import { danger } from "../globals.js";
import { SafeOpenError, openFileWithinRoot } from "../infra/fs-safe.js";
import { defaultRuntime, type RuntimeEnv } from "../runtime.js";
import { detectMime } from "./mime.js";
import { cleanOldMedia, getMediaDir, MEDIA_MAX_BYTES } from "./store.js";

const DEFAULT_TTL_MS = 2 * 60 * 1000;
const MAX_MEDIA_ID_CHARS = 200;
const MEDIA_ID_PATTERN = /^[\p{L}\p{N}._-]+$/u;
const MAX_MEDIA_BYTES = MEDIA_MAX_BYTES;
const HEALTH_PATH = "/media/__health";
const HEALTH_TOKEN_FILE = ".server-token";

const isValidMediaId = (id: string) => {
  if (!id) {
    return false;
  }
  if (id.length > MAX_MEDIA_ID_CHARS) {
    return false;
  }
  if (id === "." || id === "..") {
    return false;
  }
  return MEDIA_ID_PATTERN.test(id);
};

export function attachMediaRoutes(
  app: Express,
  ttlMs = DEFAULT_TTL_MS,
  _runtime: RuntimeEnv = defaultRuntime,
) {
  const mediaDir = getMediaDir();

  app.get(HEALTH_PATH, async (_req, res) => {
    const token = await readMediaServerToken().catch(() => null);
    if (!token) {
      res.status(503).json({ ok: false });
      return;
    }
    res.json({ ok: true, token });
  });

  app.get("/media/:id", async (req, res) => {
    const id = req.params.id;
    if (!isValidMediaId(id)) {
      res.status(400).send("invalid path");
      return;
    }
    try {
      const { handle, realPath, stat } = await openFileWithinRoot({
        rootDir: mediaDir,
        relativePath: id,
      });
      if (stat.size > MAX_MEDIA_BYTES) {
        await handle.close().catch(() => {});
        res.status(413).send("too large");
        return;
      }
      if (Date.now() - stat.mtimeMs > ttlMs) {
        await handle.close().catch(() => {});
        await fs.rm(realPath).catch(() => {});
        res.status(410).send("expired");
        return;
      }
      const data = await handle.readFile();
      await handle.close().catch(() => {});
      const mime = await detectMime({ buffer: data, filePath: realPath });
      if (mime) {
        res.type(mime);
      }
      res.send(data);
      // best-effort single-use cleanup after response ends
      res.on("finish", () => {
        const cleanup = () => {
          void fs.rm(realPath).catch(() => {});
        };
        // Tests should not pay for time-based cleanup delays.
        if (process.env.VITEST || process.env.NODE_ENV === "test") {
          queueMicrotask(cleanup);
          return;
        }
        setTimeout(cleanup, 50);
      });
    } catch (err) {
      if (err instanceof SafeOpenError) {
        if (err.code === "invalid-path") {
          res.status(400).send("invalid path");
          return;
        }
        if (err.code === "not-found") {
          res.status(404).send("not found");
          return;
        }
      }
      res.status(404).send("not found");
    }
  });

  // periodic cleanup
  setInterval(() => {
    void cleanOldMedia(ttlMs);
  }, ttlMs).unref();
}

export async function startMediaServer(
  port: number,
  ttlMs = DEFAULT_TTL_MS,
  runtime: RuntimeEnv = defaultRuntime,
): Promise<Server> {
  const app = express();
  await writeMediaServerToken(crypto.randomUUID());
  attachMediaRoutes(app, ttlMs, runtime);
  return await new Promise((resolve, reject) => {
    const server = app.listen(port, "127.0.0.1");
    server.once("listening", () => resolve(server));
    server.once("error", (err) => {
      runtime.error(danger(`Media server failed: ${String(err)}`));
      reject(err);
    });
  });
}

export function getMediaServerHealthPath() {
  return HEALTH_PATH;
}

export async function readMediaServerToken(): Promise<string | null> {
  const token = await fs.readFile(pathJoinMediaDir(HEALTH_TOKEN_FILE), "utf8").catch(() => null);
  const trimmed = token?.trim();
  return trimmed ? trimmed : null;
}

export async function probeMediaServerIdentity(
  port: number,
  expectedToken: string,
): Promise<boolean> {
  if (!expectedToken.trim()) {
    return false;
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 1_000);
  try {
    const res = await fetch(`http://127.0.0.1:${port}${HEALTH_PATH}`, {
      signal: controller.signal,
    });
    if (!res.ok) {
      return false;
    }
    const payload = (await res.json().catch(() => null)) as { token?: unknown } | null;
    return payload?.token === expectedToken;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

async function writeMediaServerToken(token: string): Promise<void> {
  await fs.mkdir(getMediaDir(), { recursive: true, mode: 0o700 });
  await fs.writeFile(pathJoinMediaDir(HEALTH_TOKEN_FILE), token, { mode: 0o600 });
}

function pathJoinMediaDir(file: string): string {
  return path.join(getMediaDir(), file);
}

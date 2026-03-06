import crypto from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import * as tar from "tar";
import { validateArchiveEntryPath } from "../infra/archive-path.js";
import { extractArchive } from "../infra/archive.js";
import type { MigrateArchiveEntry, MigrateManifest } from "./migrate-types.js";
import { parseMigrateManifest } from "./migrate-types.js";

const ENCRYPTED_SUFFIX = ".enc";
const SCRYPT_N = 16_384;
const SCRYPT_R = 8;
const SCRYPT_P = 1;
const SALT_BYTES = 32;
const IV_BYTES = 16;
const AUTH_TAG_BYTES = 16;

export function isEncryptedMigrateArchive(filePath: string): boolean {
  return filePath.toLowerCase().endsWith(ENCRYPTED_SUFFIX);
}

export async function createMigrateArchive(
  entries: readonly MigrateArchiveEntry[],
  manifest: MigrateManifest,
  outputPath: string,
): Promise<void> {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-migrate-archive-"));
  try {
    const scopesDir = path.join(tempDir, "scopes");
    await fs.mkdir(scopesDir, { recursive: true });
    await fs.writeFile(
      path.join(tempDir, "manifest.json"),
      `${JSON.stringify(manifest, null, 2)}\n`,
    );
    for (const entry of entries) {
      validateArchiveEntryPath(entry.archivePath);
      const source = path.resolve(entry.sourcePath);
      const target = path.join(tempDir, entry.archivePath);
      await fs.mkdir(path.dirname(target), { recursive: true });
      if (entry.kind === "directory") {
        await fs.cp(source, target, { recursive: true });
        continue;
      }
      await fs.copyFile(source, target);
    }
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await tar.c(
      {
        cwd: tempDir,
        file: outputPath,
        gzip: true,
        portable: true,
        noMtime: true,
      },
      ["manifest.json", "scopes"],
    );
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

export async function encryptArchive(
  inputPath: string,
  outputPath: string,
  password: string,
): Promise<void> {
  const salt = crypto.randomBytes(SALT_BYTES);
  const iv = crypto.randomBytes(IV_BYTES);
  const key = await deriveKey(password, salt);
  const cipher = crypto.createCipheriv("aes-256-gcm", key, iv);
  const output = createWriteStream(outputPath);
  const input = createReadStream(inputPath);
  output.write(salt);
  output.write(iv);
  await pipeline(input, cipher, output, { end: false });
  output.end(cipher.getAuthTag());
  await waitForStreamClose(output);
}

export async function decryptArchive(
  inputPath: string,
  outputPath: string,
  password: string,
): Promise<void> {
  const stat = await fs.stat(inputPath);
  const minimumBytes = SALT_BYTES + IV_BYTES + AUTH_TAG_BYTES;
  if (stat.size <= minimumBytes) {
    throw new Error("Encrypted archive is too small.");
  }
  const fd = await fs.open(inputPath, "r");
  try {
    const salt = Buffer.alloc(SALT_BYTES);
    const iv = Buffer.alloc(IV_BYTES);
    const authTag = Buffer.alloc(AUTH_TAG_BYTES);
    await fd.read(salt, 0, SALT_BYTES, 0);
    await fd.read(iv, 0, IV_BYTES, SALT_BYTES);
    await fd.read(authTag, 0, AUTH_TAG_BYTES, stat.size - AUTH_TAG_BYTES);
    const key = await deriveKey(password, salt);
    const decipher = crypto.createDecipheriv("aes-256-gcm", key, iv);
    decipher.setAuthTag(authTag);
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    try {
      await pipeline(
        createReadStream(inputPath, {
          start: SALT_BYTES + IV_BYTES,
          end: stat.size - AUTH_TAG_BYTES - 1,
        }),
        decipher,
        createWriteStream(outputPath),
      );
    } catch (err) {
      throw new Error("Failed to decrypt archive. Check the password and archive integrity.", {
        cause: err,
      });
    }
  } finally {
    await fd.close();
  }
}

export async function readArchiveManifest(
  archivePath: string,
  password?: string,
): Promise<MigrateManifest> {
  const extractDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-migrate-manifest-"));
  try {
    await extractMigrateArchive(archivePath, extractDir, password);
    const raw = JSON.parse(await fs.readFile(path.join(extractDir, "manifest.json"), "utf-8"));
    return parseMigrateManifest(raw);
  } finally {
    await fs.rm(extractDir, { recursive: true, force: true });
  }
}

export async function extractMigrateArchive(
  archivePath: string,
  destDir: string,
  password?: string,
): Promise<void> {
  const prepared = await withPreparedArchive(archivePath, password, async (plainArchivePath) => {
    await fs.mkdir(destDir, { recursive: true });
    await extractArchive({
      archivePath: plainArchivePath,
      destDir,
      timeoutMs: 120_000,
      kind: "tar",
      tarGzip: true,
    });
  });
  return prepared;
}

async function withPreparedArchive<T>(
  archivePath: string,
  password: string | undefined,
  run: (plainArchivePath: string) => Promise<T>,
): Promise<T> {
  if (!isEncryptedMigrateArchive(archivePath)) {
    return await run(archivePath);
  }
  if (!password) {
    throw new Error("Encrypted archive requires a password.");
  }
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-migrate-decrypt-"));
  const plainArchivePath = path.join(tempDir, "archive.tar.gz");
  try {
    await decryptArchive(archivePath, plainArchivePath, password);
    return await run(plainArchivePath);
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

async function deriveKey(password: string, salt: Buffer): Promise<Buffer> {
  return await new Promise<Buffer>((resolve, reject) => {
    crypto.scrypt(password, salt, 32, { N: SCRYPT_N, r: SCRYPT_R, p: SCRYPT_P }, (err, key) => {
      if (err) {
        reject(err);
        return;
      }
      resolve(Buffer.from(key));
    });
  });
}

async function waitForStreamClose(stream: NodeJS.WritableStream): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    stream.once("close", () => resolve());
    stream.once("error", reject);
  });
}

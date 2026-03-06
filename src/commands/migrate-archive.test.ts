import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  createMigrateArchive,
  encryptArchive,
  extractMigrateArchive,
  readArchiveManifest,
} from "./migrate-archive.js";
import { createMigrateManifest, resolveMigrateArchivePath } from "./migrate-types.js";

let fixtureRoot = "";
let fixtureCount = 0;

async function makeTempDir(prefix = "case") {
  const dir = path.join(fixtureRoot, `${prefix}-${fixtureCount++}`);
  await fs.mkdir(dir, { recursive: true });
  return dir;
}

beforeAll(async () => {
  fixtureRoot = await fs.mkdtemp(path.join(os.tmpdir(), "marv-migrate-archive-"));
});

afterAll(async () => {
  await fs.rm(fixtureRoot, { recursive: true, force: true });
});

describe("migrate archive", () => {
  it("creates a plain archive, reads the manifest, and extracts files", async () => {
    const workDir = await makeTempDir();
    const sourceDir = path.join(workDir, "source");
    const archivePath = path.join(workDir, "export.tar.gz");
    const extractDir = path.join(workDir, "extract");

    await fs.mkdir(path.join(sourceDir, "memory", "soul"), { recursive: true });
    await fs.writeFile(path.join(sourceDir, "memory", "soul", "main.sqlite"), "db-bytes");

    const entries = [
      {
        scope: "memory" as const,
        sourcePath: path.join(sourceDir, "memory", "soul"),
        archivePath: resolveMigrateArchivePath({ scope: "memory", relativePath: "soul" }),
        kind: "directory" as const,
      },
    ];
    const manifest = createMigrateManifest({
      scopes: ["memory"],
      format: "plain",
      marvVersion: "0.0.0-test",
      items: entries.map(({ scope, archivePath, kind }) => ({ scope, archivePath, kind })),
    });

    await createMigrateArchive(entries, manifest, archivePath);

    const parsedManifest = await readArchiveManifest(archivePath);
    expect(parsedManifest.scopes).toEqual(["memory"]);
    expect(parsedManifest.items[0]?.archivePath).toBe("scopes/memory/soul");

    await extractMigrateArchive(archivePath, extractDir);
    const extracted = await fs.readFile(
      path.join(extractDir, "scopes", "memory", "soul", "main.sqlite"),
      "utf-8",
    );
    expect(extracted).toBe("db-bytes");
  });

  it("supports encrypted archives", async () => {
    const workDir = await makeTempDir();
    const sourceFile = path.join(workDir, "config.json");
    const plainArchivePath = path.join(workDir, "export.tar.gz");
    const encryptedArchivePath = `${plainArchivePath}.enc`;
    const extractDir = path.join(workDir, "extract");

    await fs.writeFile(sourceFile, '{"name":"marv"}\n');

    const entries = [
      {
        scope: "config" as const,
        sourcePath: sourceFile,
        archivePath: resolveMigrateArchivePath({ scope: "config", relativePath: "marv.json" }),
        kind: "file" as const,
      },
    ];
    const manifest = createMigrateManifest({
      scopes: ["config"],
      format: "encrypted",
      marvVersion: "0.0.0-test",
      items: entries.map(({ scope, archivePath, kind }) => ({ scope, archivePath, kind })),
    });

    await createMigrateArchive(entries, manifest, plainArchivePath);
    await encryptArchive(plainArchivePath, encryptedArchivePath, "secret-passphrase");

    const parsedManifest = await readArchiveManifest(encryptedArchivePath, "secret-passphrase");
    expect(parsedManifest.format).toBe("encrypted");

    await extractMigrateArchive(encryptedArchivePath, extractDir, "secret-passphrase");
    const extracted = await fs.readFile(
      path.join(extractDir, "scopes", "config", "marv.json"),
      "utf-8",
    );
    expect(extracted).toContain('"name":"marv"');
  });
});

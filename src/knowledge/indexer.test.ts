import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { querySoulMemoryMulti } from "../memory/storage/soul-memory-store.js";
import { indexDirectory } from "./indexer.js";

const ORIGINAL_STATE_DIR = process.env.MARV_STATE_DIR;

let stateDir = "";
let vaultDir = "";

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-knowledge-state-"));
  vaultDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-knowledge-vault-"));
  process.env.MARV_STATE_DIR = stateDir;
});

afterEach(async () => {
  if (ORIGINAL_STATE_DIR === undefined) {
    delete process.env.MARV_STATE_DIR;
  } else {
    process.env.MARV_STATE_DIR = ORIGINAL_STATE_DIR;
  }
  await fs.rm(stateDir, { recursive: true, force: true });
  await fs.rm(vaultDir, { recursive: true, force: true });
});

describe("knowledge indexer", () => {
  it("indexes markdown files into document-scoped soul memory and skips unchanged files", async () => {
    await fs.mkdir(path.join(vaultDir, "Projects"), { recursive: true });
    await fs.writeFile(
      path.join(vaultDir, "Projects", "marv.md"),
      `---
tags:
  - project
---

# Marv

## Setup Guide

Install Node 22 and run pnpm install.
`,
      "utf-8",
    );

    const first = await indexDirectory({
      agentId: "main",
      directory: vaultDir,
      vaultName: "Vault",
    });
    expect(first.filesIndexed).toBe(1);
    expect(first.chunksWritten).toBeGreaterThan(0);

    const results = querySoulMemoryMulti({
      agentId: "main",
      scopes: [{ scopeType: "agent", scopeId: "main", weight: 1 }],
      query: "Node 22 setup",
      topK: 5,
      minScore: 0,
    });
    const doc = results.find((entry) => entry.scopeType === "document");
    expect(doc?.metadata?.relativePath).toBe("Projects/marv.md");
    expect(doc?.metadata?.heading).toBe("## Setup Guide");

    const second = await indexDirectory({
      agentId: "main",
      directory: vaultDir,
      vaultName: "Vault",
    });
    expect(second.filesSkipped).toBe(1);
  });
});

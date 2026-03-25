import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readSkillUsageRecords } from "../../skill-usage-records.js";
import { persistSynthesizedTool } from "./tool-synthesis.js";

const ORIGINAL_STATE_DIR = process.env.MARV_STATE_DIR;
const ORIGINAL_HOME = process.env.HOME;

let stateDir = "";
let tempDir = "";

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-tool-synth-state-"));
  tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-tool-synth-work-"));
  process.env.MARV_STATE_DIR = stateDir;
  process.env.HOME = tempDir;
});

afterEach(async () => {
  if (ORIGINAL_STATE_DIR === undefined) {
    delete process.env.MARV_STATE_DIR;
  } else {
    process.env.MARV_STATE_DIR = ORIGINAL_STATE_DIR;
  }
  if (ORIGINAL_HOME === undefined) {
    delete process.env.HOME;
  } else {
    process.env.HOME = ORIGINAL_HOME;
  }
  await fs.rm(stateDir, { recursive: true, force: true });
  await fs.rm(tempDir, { recursive: true, force: true });
});

describe("persistSynthesizedTool", () => {
  it("creates a managed skill with script and usage record", async () => {
    const scriptPath = path.join(tempDir, "probe.sh");
    await fs.writeFile(scriptPath, "#!/bin/bash\necho ok\n", "utf8");

    const result = await persistSynthesizedTool({
      name: "parquet-inspector",
      description: "Inspect parquet metadata from the command line.",
      scriptPath,
      managedSkillsDir: path.join(stateDir, "skills"),
    });

    expect(result.ok).toBe(true);
    const skillDir = path.join(stateDir, "skills", "parquet-inspector");
    const skillDoc = await fs.readFile(path.join(skillDir, "SKILL.md"), "utf8");
    const copiedScript = await fs.readFile(path.join(skillDir, "scripts", "probe.sh"), "utf8");
    const records = await readSkillUsageRecords(
      path.join(tempDir, ".marv", "skills", ".usage-records.json"),
    );

    expect(skillDoc).toContain("name: parquet-inspector");
    expect(skillDoc).toContain("Inspect parquet metadata from the command line.");
    expect(skillDoc).toContain("scripts/probe.sh");
    expect(copiedScript).toContain("echo ok");
    expect(records["parquet-inspector"]?.skillId).toBe("parquet-inspector");
  });
});

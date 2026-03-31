import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readSkillUsageRecords } from "../skill-usage-records.js";
import { createSkillCrystallizeTool } from "./skill-crystallize-tool.js";
import { createSkillViewTool } from "./skill-view-tool.js";

const tempDirs: string[] = [];
let originalHome: string | undefined;

async function makeTempDir(prefix: string) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

async function writeSkill(params: {
  dir: string;
  name: string;
  description: string;
  body?: string;
  files?: Array<{ path: string; content: string }>;
}) {
  await fs.mkdir(params.dir, { recursive: true });
  await fs.writeFile(
    path.join(params.dir, "SKILL.md"),
    `---\nname: ${params.name}\ndescription: ${params.description}\n---\n\n${params.body ?? "# Body\n"}`,
    "utf-8",
  );
  for (const file of params.files ?? []) {
    const destination = path.join(params.dir, file.path);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.writeFile(destination, file.content, "utf-8");
  }
}

beforeEach(async () => {
  originalHome = process.env.HOME;
  process.env.HOME = await makeTempDir("marv-skill-home-");
});

afterEach(async () => {
  if (originalHome === undefined) {
    delete process.env.HOME;
  } else {
    process.env.HOME = originalHome;
  }
  await Promise.all(
    tempDirs.splice(0, tempDirs.length).map((dir) => fs.rm(dir, { recursive: true, force: true })),
  );
});

describe("skill_view", () => {
  it("loads stripped skill instructions and bundled files", async () => {
    const workspaceDir = await makeTempDir("marv-skill-workspace-");
    await writeSkill({
      dir: path.join(workspaceDir, "skills", "demo"),
      name: "demo",
      description: "Demo skill",
      body: "## Overview\n\nUse it.\n",
      files: [{ path: "references/guide.md", content: "Guide body\n" }],
    });

    const tool = createSkillViewTool({ workspaceDir });
    const skillResult = await tool.execute("call1", { name: "demo" });
    expect(skillResult.details).toMatchObject({
      ok: true,
      name: "demo",
    });
    expect((skillResult.details as { content: string }).content).toContain("## Overview");
    expect((skillResult.details as { content: string }).content).not.toContain("description:");

    const fileResult = await tool.execute("call2", { name: "demo", file: "references/guide.md" });
    expect((fileResult.details as { content: string }).content).toContain("Guide body");

    const records = await readSkillUsageRecords();
    expect(records.demo?.firstUsedAt).toBeTypeOf("number");
    expect(records.demo?.lastUsedAt).toBeTypeOf("number");
  });

  it("rejects path traversal when loading bundled resources", async () => {
    const workspaceDir = await makeTempDir("marv-skill-workspace-");
    await writeSkill({
      dir: path.join(workspaceDir, "skills", "demo"),
      name: "demo",
      description: "Demo skill",
    });

    const tool = createSkillViewTool({ workspaceDir });
    await expect(tool.execute("call3", { name: "demo", file: "../../etc/passwd" })).rejects.toThrow(
      "file must stay within the skill directory",
    );
  });
});

describe("skill_crystallize", () => {
  it("rewrites the canonical skill and records validated success", async () => {
    const workspaceDir = await makeTempDir("marv-skill-workspace-");
    const skillDir = path.join(workspaceDir, "skills", "demo");
    await writeSkill({
      dir: skillDir,
      name: "demo",
      description: "Original skill",
      body: "## Overview\n\nOriginal.\n",
    });

    const tool = createSkillCrystallizeTool({ workspaceDir });
    const result = await tool.execute("call4", {
      name: "demo",
      description: "Improved skill",
      body: "## Overview\n\nImproved.\n\n## Workflow\n\n1. Run it.\n",
      sourceSkill: "demo",
      validatedSuccess: true,
      files: [{ path: "references/usage.md", content: "Usage notes" }],
    });

    expect(result.details).toMatchObject({
      ok: true,
      name: "demo",
      replaced: true,
    });

    const skillMd = await fs.readFile(path.join(skillDir, "SKILL.md"), "utf-8");
    expect(skillMd).toContain("Improved skill");
    expect(skillMd).toContain("Improved.");
    expect(await fs.readFile(path.join(skillDir, "references", "usage.md"), "utf-8")).toContain(
      "Usage notes",
    );

    const records = await readSkillUsageRecords();
    expect(records.demo?.successCount).toBe(1);
    expect(records.demo?.lastOutcome).toBe("success");
    expect(records.demo?.lastValidatedAt).toBeTypeOf("number");
  });
});

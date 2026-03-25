import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createCliInvokeTool } from "./cli-invoke-tool.js";
import { createCliProfilesTool } from "./cli-profiles-tool.js";
import { createCliSynthesizeTool } from "./cli-synthesize-tool.js";
import { createCliVerifyTool } from "./cli-verify-tool.js";

const ORIGINAL_STATE_DIR = process.env.MARV_STATE_DIR;

let stateDir = "";
let workspaceDir = "";

beforeEach(async () => {
  stateDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-cli-tools-state-"));
  workspaceDir = await fs.mkdtemp(path.join(os.tmpdir(), "marv-cli-tools-work-"));
  process.env.MARV_STATE_DIR = stateDir;
});

afterEach(async () => {
  if (ORIGINAL_STATE_DIR === undefined) {
    delete process.env.MARV_STATE_DIR;
  } else {
    process.env.MARV_STATE_DIR = ORIGINAL_STATE_DIR;
  }
  await fs.rm(stateDir, { recursive: true, force: true }).catch(() => {});
  await fs.rm(workspaceDir, { recursive: true, force: true }).catch(() => {});
});

describe("managed CLI synthesis tools", () => {
  it("synthesizes an active script-wrapper profile and invokes it", async () => {
    const scriptPath = path.join(workspaceDir, "hello-cli.sh");
    await fs.writeFile(
      scriptPath,
      `#!/bin/bash
if [ "$1" = "--help" ]; then
  echo "usage: hello-cli"
  exit 0
fi
printf '{"ok":true,"argc":%d,"first":"%s"}\n' "$#" "\${1:-}"
`,
      "utf8",
    );
    await fs.chmod(scriptPath, 0o755);

    const synthesizeTool = createCliSynthesizeTool({ workspaceDir });
    expect(synthesizeTool).not.toBeNull();
    const synthesize = await synthesizeTool!.execute("call1", {
      id: "hello-cli",
      description: "A small synthesized CLI used for testing.",
      scriptPath,
      outputMode: "json",
      activate: true,
      capabilities: ["test.echo"],
    });
    expect(synthesize.details).toMatchObject({
      ok: true,
      profileId: "hello-cli",
      state: "active",
    });

    const invokeTool = createCliInvokeTool({ workspaceDir });
    expect(invokeTool).not.toBeNull();
    const invoke = await invokeTool!.execute("call2", {
      profileId: "hello-cli",
      extraArgs: ["hello"],
    });
    expect(invoke.details).toMatchObject({
      profileId: "hello-cli",
      status: "ok",
      parsed: {
        ok: true,
        argc: 1,
        first: "hello",
      },
    });
  });

  it("lists, inspects, disables, and re-enables synthesized profiles", async () => {
    const scriptPath = path.join(workspaceDir, "status-cli.sh");
    await fs.writeFile(
      scriptPath,
      `#!/bin/bash
if [ "$1" = "--help" ]; then
  echo "usage: status-cli"
  exit 0
fi
echo "ok"
`,
      "utf8",
    );
    await fs.chmod(scriptPath, 0o755);

    const synthesizeTool = createCliSynthesizeTool({ workspaceDir });
    await synthesizeTool!.execute("call3", {
      id: "status-cli",
      description: "Profile lifecycle test CLI.",
      scriptPath,
      outputMode: "text",
      activate: true,
      capabilities: ["test.status"],
    });

    const profilesTool = createCliProfilesTool();
    expect(profilesTool).not.toBeNull();

    const list = await profilesTool!.execute("call4", { action: "list" });
    expect(list.details).toMatchObject({
      ok: true,
    });
    expect((list.details as { profiles: Array<{ id: string }> }).profiles).toEqual(
      expect.arrayContaining([expect.objectContaining({ id: "status-cli" })]),
    );

    const inspect = await profilesTool!.execute("call5", {
      action: "inspect",
      profileId: "status-cli",
    });
    expect(inspect.details).toMatchObject({
      ok: true,
      profile: {
        entry: {
          id: "status-cli",
          state: "active",
        },
      },
    });

    const disable = await profilesTool!.execute("call6", {
      action: "disable",
      profileId: "status-cli",
    });
    expect(disable.details).toMatchObject({
      ok: true,
      profileId: "status-cli",
      state: "verified",
    });

    const enable = await profilesTool!.execute("call7", {
      action: "enable",
      profileId: "status-cli",
    });
    expect(enable.details).toMatchObject({
      ok: true,
      profileId: "status-cli",
      state: "active",
    });
  });

  it("can quarantine a failing profile during verification", async () => {
    const synthesizeTool = createCliSynthesizeTool({ workspaceDir });
    const synthesized = await synthesizeTool!.execute("call8", {
      id: "broken-cli",
      description: "Broken command test profile.",
      command: "definitely-missing-marv-command",
      outputMode: "text",
      activate: false,
      skipHelpCheck: true,
    });
    expect(synthesized.details).toMatchObject({
      ok: false,
      profileId: "broken-cli",
      state: "draft",
    });

    const verifyTool = createCliVerifyTool({ workspaceDir });
    const verified = await verifyTool!.execute("call9", {
      profileId: "broken-cli",
      quarantineOnFail: true,
    });
    expect(verified.details).toMatchObject({
      ok: false,
      profileId: "broken-cli",
      state: "quarantined",
    });
  });

  it("keeps an active profile active after re-verification", async () => {
    const scriptPath = path.join(workspaceDir, "verify-cli.sh");
    await fs.writeFile(
      scriptPath,
      `#!/bin/bash
if [ "$1" = "--help" ]; then
  echo "usage: verify-cli"
  exit 0
fi
echo "ready"
`,
      "utf8",
    );
    await fs.chmod(scriptPath, 0o755);

    const synthesizeTool = createCliSynthesizeTool({ workspaceDir });
    await synthesizeTool!.execute("call10", {
      id: "verify-cli",
      description: "Active profile verification state test.",
      scriptPath,
      outputMode: "text",
      activate: true,
    });

    const verifyTool = createCliVerifyTool({ workspaceDir });
    const verified = await verifyTool!.execute("call11", {
      profileId: "verify-cli",
    });
    expect(verified.details).toMatchObject({
      ok: true,
      profileId: "verify-cli",
      state: "active",
      verification: {
        helpExitCode: 0,
      },
    });
  });
});

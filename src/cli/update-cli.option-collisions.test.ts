import { Command } from "commander";
import { beforeEach, describe, expect, it, vi } from "vitest";

const updateCommand = vi.fn(async (_opts: unknown) => {});
const updateRollbackCommand = vi.fn(async (_opts: unknown) => {});
const updateStatusCommand = vi.fn(async (_opts: unknown) => {});
const updateWizardCommand = vi.fn(async (_opts: unknown) => {});

const defaultRuntime = {
  log: vi.fn(),
  error: vi.fn(),
  exit: vi.fn(),
};

vi.mock("./update-cli/update-command.js", () => ({
  updateCommand: (opts: unknown) => updateCommand(opts),
}));

vi.mock("./update-cli/rollback.js", () => ({
  updateRollbackCommand: (opts: unknown) => updateRollbackCommand(opts),
}));

vi.mock("./update-cli/status.js", () => ({
  updateStatusCommand: (opts: unknown) => updateStatusCommand(opts),
}));

vi.mock("./update-cli/wizard.js", () => ({
  updateWizardCommand: (opts: unknown) => updateWizardCommand(opts),
}));

vi.mock("../runtime.js", () => ({
  defaultRuntime,
}));

describe("update cli option collisions", () => {
  beforeEach(() => {
    updateCommand.mockClear();
    updateRollbackCommand.mockClear();
    updateStatusCommand.mockClear();
    updateWizardCommand.mockClear();
    defaultRuntime.log.mockClear();
    defaultRuntime.error.mockClear();
    defaultRuntime.exit.mockClear();
  });

  it("forwards parent-captured --json/--timeout to `update status`", async () => {
    const { registerUpdateCli } = await import("./update-cli.js");
    const program = new Command();
    registerUpdateCli(program);

    await program.parseAsync(["update", "status", "--json", "--timeout", "9"], { from: "user" });

    expect(updateStatusCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        json: true,
        timeout: "9",
      }),
    );
  });

  it("forwards parent-captured --json/--timeout to `update rollback`", async () => {
    const { registerUpdateCli } = await import("./update-cli.js");
    const program = new Command();
    registerUpdateCli(program);

    await program.parseAsync(["update", "rollback", "--json", "--timeout", "11"], {
      from: "user",
    });

    expect(updateRollbackCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        json: true,
        timeout: "11",
      }),
    );
  });

  it("forwards parent-captured --timeout to `update wizard`", async () => {
    const { registerUpdateCli } = await import("./update-cli.js");
    const program = new Command();
    registerUpdateCli(program);

    await program.parseAsync(["update", "wizard", "--timeout", "13"], { from: "user" });

    expect(updateWizardCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        timeout: "13",
      }),
    );
  });
});

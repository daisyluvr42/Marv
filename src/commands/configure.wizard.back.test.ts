import { describe, expect, it, vi } from "vitest";
import type { MarvConfig } from "../core/config/config.js";

const mocks = vi.hoisted(() => ({
  clackIntro: vi.fn(),
  clackOutro: vi.fn(),
  clackSelect: vi.fn(),
  clackText: vi.fn(),
  clackConfirm: vi.fn(),
  readConfigFileSnapshot: vi.fn(),
  writeConfigFile: vi.fn(),
  resolveGatewayPort: vi.fn(),
  ensureControlUiAssetsBuilt: vi.fn(),
  createClackPrompter: vi.fn(),
  note: vi.fn(),
  printWizardHeader: vi.fn(),
  probeGatewayReachable: vi.fn(),
  waitForGatewayReachable: vi.fn(),
  resolveControlUiLinks: vi.fn(),
  summarizeExistingConfig: vi.fn(),
  promptGatewayConfig: vi.fn(),
  promptAuthConfig: vi.fn(),
}));

vi.mock("@clack/prompts", () => ({
  intro: mocks.clackIntro,
  outro: mocks.clackOutro,
  select: mocks.clackSelect,
  text: mocks.clackText,
  confirm: mocks.clackConfirm,
}));

vi.mock("../core/config/config.js", () => ({
  CONFIG_PATH: "~/.marv/marv.json",
  readConfigFileSnapshot: mocks.readConfigFileSnapshot,
  writeConfigFile: mocks.writeConfigFile,
  resolveGatewayPort: mocks.resolveGatewayPort,
}));

vi.mock("../infra/control-ui-assets.js", () => ({
  ensureControlUiAssetsBuilt: mocks.ensureControlUiAssetsBuilt,
}));

vi.mock("../wizard/clack-prompter.js", () => ({
  createClackPrompter: mocks.createClackPrompter,
}));

vi.mock("../terminal/note.js", () => ({
  note: mocks.note,
}));

vi.mock("./onboard-helpers.js", () => ({
  DEFAULT_WORKSPACE: "~/.marv/workspace",
  applyWizardMetadata: (cfg: MarvConfig) => cfg,
  ensureWorkspaceAndSessions: vi.fn(),
  guardCancel: <T>(value: T) => value,
  printWizardHeader: mocks.printWizardHeader,
  probeGatewayReachable: mocks.probeGatewayReachable,
  resolveControlUiLinks: mocks.resolveControlUiLinks,
  summarizeExistingConfig: mocks.summarizeExistingConfig,
  waitForGatewayReachable: mocks.waitForGatewayReachable,
}));

vi.mock("./health.js", () => ({ healthCommand: vi.fn() }));
vi.mock("./health-format.js", () => ({ formatHealthCheckFailure: vi.fn() }));
vi.mock("./configure.gateway.js", () => ({
  promptGatewayConfig: (...args: unknown[]) => mocks.promptGatewayConfig(...args),
}));
vi.mock("./configure.gateway-auth.js", () => ({
  promptAuthConfig: (...args: unknown[]) => mocks.promptAuthConfig(...args),
}));
vi.mock("./configure.channels.js", () => ({ removeChannelConfigWizard: vi.fn() }));
vi.mock("./configure.daemon.js", () => ({ maybeInstallDaemon: vi.fn() }));
vi.mock("./onboard-remote.js", () => ({ promptRemoteGatewayConfig: vi.fn() }));
vi.mock("./onboard-skills.js", () => ({ setupSkills: vi.fn() }));
vi.mock("./onboard-channels.js", () => ({ setupChannels: vi.fn() }));

import { afterEach } from "vitest";
import { WizardBackSignal } from "../wizard/prompts.js";
import { runConfigureWizard } from "./configure.wizard.js";

function setupCommonMocks() {
  mocks.readConfigFileSnapshot.mockResolvedValue({
    exists: true,
    valid: true,
    config: { gateway: { mode: "local" } },
    issues: [],
  });
  mocks.resolveGatewayPort.mockReturnValue(4242);
  mocks.probeGatewayReachable.mockResolvedValue({ ok: false });
  mocks.resolveControlUiLinks.mockReturnValue({ wsUrl: "ws://127.0.0.1:4242" });
  mocks.summarizeExistingConfig.mockReturnValue("");
  mocks.createClackPrompter.mockReturnValue({});
  mocks.clackIntro.mockResolvedValue(undefined);
  mocks.clackOutro.mockResolvedValue(undefined);
  mocks.clackText.mockResolvedValue("");
  mocks.clackConfirm.mockResolvedValue(false);
  mocks.ensureControlUiAssetsBuilt.mockResolvedValue({ ok: true });
}

describe("configure wizard batched back-navigation", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("gracefully exits without persisting when Back is pressed on the first section", async () => {
    setupCommonMocks();
    const runtime = { log: vi.fn(), error: vi.fn(), exit: vi.fn() };

    // Gateway section throws WizardBackSignal (user pressed Back).
    mocks.promptGatewayConfig.mockRejectedValueOnce(new WizardBackSignal());

    await runConfigureWizard({ command: "configure", sections: ["gateway"] }, runtime);

    expect(mocks.writeConfigFile).not.toHaveBeenCalled();
    expect(runtime.exit).not.toHaveBeenCalled();
    expect(mocks.clackOutro).toHaveBeenCalledWith("No changes applied.");
  });

  it("goes back to the first section when Back is pressed on the second section", async () => {
    setupCommonMocks();
    const runtime = { log: vi.fn(), error: vi.fn(), exit: vi.fn() };

    const authCfg = { gateway: { mode: "local" }, model: { provider: "test" } } as MarvConfig;
    // Model section: completes both times.
    mocks.promptAuthConfig.mockResolvedValueOnce(authCfg).mockResolvedValueOnce(authCfg);

    // Gateway section: first call Back, second call succeeds.
    mocks.promptGatewayConfig
      .mockRejectedValueOnce(new WizardBackSignal())
      .mockResolvedValueOnce({ config: authCfg, port: 4242, token: undefined });

    await runConfigureWizard({ command: "configure", sections: ["model", "gateway"] }, runtime);

    // Model ran twice (initial + after back).
    expect(mocks.promptAuthConfig).toHaveBeenCalledTimes(2);
    // Gateway ran twice (back + success).
    expect(mocks.promptGatewayConfig).toHaveBeenCalledTimes(2);
    // Config persisted after all sections completed.
    expect(mocks.writeConfigFile).toHaveBeenCalled();
    expect(runtime.exit).not.toHaveBeenCalled();
  });

  it("does not persist config when Back cancels the only section", async () => {
    setupCommonMocks();
    const runtime = { log: vi.fn(), error: vi.fn(), exit: vi.fn() };

    mocks.promptAuthConfig.mockRejectedValueOnce(new WizardBackSignal());

    await runConfigureWizard({ command: "configure", sections: ["model"] }, runtime);

    expect(mocks.writeConfigFile).not.toHaveBeenCalled();
    expect(runtime.exit).not.toHaveBeenCalled();
  });
});

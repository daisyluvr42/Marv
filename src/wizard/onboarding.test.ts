import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_BOOTSTRAP_FILENAME } from "../agents/workspace.js";
import type { RuntimeEnv } from "../runtime.js";
import { runOnboardingWizard } from "./onboarding.js";
import type { WizardPrompter, WizardSelectParams } from "./prompts.js";

const ensureAuthProfileStore = vi.hoisted(() => vi.fn(() => ({ profiles: {} })));
const promptAuthChoiceGrouped = vi.hoisted(() => vi.fn(async () => "skip"));
const applyAuthChoice = vi.hoisted(() => vi.fn(async (args) => ({ config: args.config })));
const resolvePreferredProviderForAuthChoice = vi.hoisted(() => vi.fn(() => "openai"));
const warnIfModelConfigLooksOff = vi.hoisted(() => vi.fn(async () => {}));
const applyPrimaryModel = vi.hoisted(() => vi.fn((cfg) => cfg));
const promptDefaultModel = vi.hoisted(() => vi.fn(async () => ({ config: null, model: null })));
const runAuthProbes = vi.hoisted(() =>
  vi.fn(async () => ({
    startedAt: 0,
    finishedAt: 1,
    durationMs: 1,
    totalTargets: 1,
    options: {
      provider: "openai",
      timeoutMs: 8000,
      concurrency: 1,
      maxTokens: 16,
    },
    results: [
      {
        provider: "openai",
        model: "openai/gpt-5.2",
        label: "default",
        source: "env",
        status: "ok",
      },
    ],
  })),
);
const describeProbeSummary = vi.hoisted(() => vi.fn(() => "Probed 1 target in 1ms"));
const promptCustomApiConfig = vi.hoisted(() => vi.fn(async (args) => ({ config: args.config })));
const configureGatewayForOnboarding = vi.hoisted(() =>
  vi.fn(async (args) => ({
    nextConfig: args.nextConfig,
    settings: {
      port: args.localPort ?? 18789,
      bind: "loopback",
      authMode: "token",
      gatewayToken: "test-token",
      tailscaleMode: "off",
      tailscaleResetOnExit: false,
    },
  })),
);
const finalizeOnboardingWizard = vi.hoisted(() =>
  vi.fn(async (options) => {
    if (!process.env.BRAVE_API_KEY) {
      await options.prompter.note("hint", "Web search (optional)");
    }

    if (options.opts.skipUi) {
      return { launchedTui: false };
    }

    const hatch = await options.prompter.select({
      message: "How do you want to hatch your bot?",
      options: [],
    });
    if (hatch !== "tui") {
      return { launchedTui: false };
    }

    let message: string | undefined;
    try {
      await fs.stat(path.join(options.workspaceDir, DEFAULT_BOOTSTRAP_FILENAME));
      message = "Wake up, my friend!";
    } catch {
      message = undefined;
    }

    await runTui({ deliver: false, message });
    return { launchedTui: true };
  }),
);
const listChannelPlugins = vi.hoisted(() => vi.fn(() => []));
const logConfigUpdated = vi.hoisted(() => vi.fn(() => {}));

const setupChannels = vi.hoisted(() => vi.fn(async (cfg) => cfg));
const setupSkills = vi.hoisted(() => vi.fn(async (cfg) => cfg));
const healthCommand = vi.hoisted(() => vi.fn(async () => {}));
const ensureWorkspaceAndSessions = vi.hoisted(() => vi.fn(async () => {}));
const writeConfigFile = vi.hoisted(() => vi.fn(async () => {}));
const readConfigFileSnapshot = vi.hoisted(() =>
  vi.fn(async () => ({
    path: "/tmp/.marv/marv.json",
    exists: false,
    raw: null as string | null,
    parsed: {},
    resolved: {},
    valid: true,
    config: {},
    issues: [] as Array<{ path: string; message: string }>,
    warnings: [] as Array<{ path: string; message: string }>,
    legacyIssues: [] as Array<{ path: string; message: string }>,
  })),
);
const ensureSystemdUserLingerInteractive = vi.hoisted(() => vi.fn(async () => {}));
const isSystemdUserServiceAvailable = vi.hoisted(() => vi.fn(async () => true));
const ensureControlUiAssetsBuilt = vi.hoisted(() => vi.fn(async () => ({ ok: true })));
const runTui = vi.hoisted(() => vi.fn(async (_options: unknown) => {}));
const setupOnboardingShellCompletion = vi.hoisted(() => vi.fn(async () => {}));

vi.mock("../commands/onboard-channels.js", () => ({
  setupChannels,
}));

vi.mock("../commands/onboard-skills.js", () => ({
  setupSkills,
}));

vi.mock("../agents/auth-profiles.js", () => ({
  ensureAuthProfileStore,
}));

vi.mock("../commands/auth-choice-prompt.js", () => ({
  promptAuthChoiceGrouped,
}));

vi.mock("../commands/auth-choice.js", () => ({
  applyAuthChoice,
  resolvePreferredProviderForAuthChoice,
  warnIfModelConfigLooksOff,
}));

vi.mock("../commands/model-picker.js", () => ({
  applyPrimaryModel,
  promptDefaultModel,
}));

vi.mock("../commands/models/list.probe.js", () => ({
  runAuthProbes,
  describeProbeSummary,
}));

vi.mock("../commands/onboard-custom.js", () => ({
  promptCustomApiConfig,
}));

vi.mock("../commands/health.js", () => ({
  healthCommand,
}));

vi.mock("../core/config/config.js", () => ({
  DEFAULT_GATEWAY_PORT: 18789,
  resolveGatewayPort: () => 18789,
  readConfigFileSnapshot,
  writeConfigFile,
}));

vi.mock("../commands/onboard-helpers.js", () => ({
  DEFAULT_WORKSPACE: "/tmp/marv-workspace",
  applyWizardMetadata: (cfg: unknown) => cfg,
  summarizeExistingConfig: () => "summary",
  handleReset: async () => {},
  randomToken: () => "test-token",
  normalizeGatewayTokenInput: (value: unknown) => ({
    ok: true,
    token: typeof value === "string" ? value.trim() : "",
    error: null,
  }),
  validateGatewayPasswordInput: () => ({ ok: true, error: null }),
  ensureWorkspaceAndSessions,
  detectBrowserOpenSupport: vi.fn(async () => ({ ok: false })),
  openUrl: vi.fn(async () => true),
  printWizardHeader: vi.fn(),
  probeGatewayReachable: vi.fn(async () => ({ ok: true })),
  waitForGatewayReachable: vi.fn(async () => {}),
  formatControlUiSshHint: vi.fn(() => "ssh hint"),
  resolveControlUiLinks: vi.fn(() => ({
    httpUrl: "http://127.0.0.1:18789",
    wsUrl: "ws://127.0.0.1:18789",
  })),
}));

vi.mock("../commands/systemd-linger.js", () => ({
  ensureSystemdUserLingerInteractive,
}));

vi.mock("../infra/daemon/systemd.js", () => ({
  isSystemdUserServiceAvailable,
}));

vi.mock("../infra/control-ui-assets.js", () => ({
  ensureControlUiAssetsBuilt,
}));

vi.mock("../channels/plugins/index.js", () => ({
  listChannelPlugins,
}));

vi.mock("../core/config/logging.js", () => ({
  logConfigUpdated,
}));

vi.mock("../tui/tui.js", () => ({
  runTui,
}));

vi.mock("./onboarding.gateway-config.js", () => ({
  configureGatewayForOnboarding,
}));

vi.mock("./onboarding.finalize.js", () => ({
  finalizeOnboardingWizard,
}));

vi.mock("./onboarding.completion.js", () => ({
  setupOnboardingShellCompletion,
}));

function createWizardPrompter(overrides?: Partial<WizardPrompter>): WizardPrompter {
  const select = vi.fn(
    async (_params: WizardSelectParams<unknown>) => "quickstart",
  ) as unknown as WizardPrompter["select"];
  return {
    intro: vi.fn(async () => {}),
    outro: vi.fn(async () => {}),
    note: vi.fn(async () => {}),
    select,
    multiselect: vi.fn(async () => []),
    text: vi.fn(async () => ""),
    confirm: vi.fn(async () => false),
    progress: vi.fn(() => ({ update: vi.fn(), stop: vi.fn() })),
    ...overrides,
  };
}

function createRuntime(opts?: { throwsOnExit?: boolean }): RuntimeEnv {
  if (opts?.throwsOnExit) {
    return {
      log: vi.fn(),
      error: vi.fn(),
      exit: vi.fn((code: number) => {
        throw new Error(`exit:${code}`);
      }),
    };
  }

  return {
    log: vi.fn(),
    error: vi.fn(),
    exit: vi.fn(),
  };
}

function parseWizardMetricsLog(runtime: RuntimeEnv): {
  interactions: number;
  total: number;
} {
  const calls = (runtime.log as ReturnType<typeof vi.fn>).mock.calls;
  const line = calls
    .map((call) => call[0])
    .find(
      (entry): entry is string => typeof entry === "string" && entry.includes("Wizard metrics:"),
    );
  if (!line) {
    throw new Error("Wizard metrics log not found");
  }
  const interactions = Number(line.match(/interactions=(\d+)/)?.[1] ?? "NaN");
  const total = Number(line.match(/total=(\d+)/)?.[1] ?? "NaN");
  return { interactions, total };
}

function getWizardNoteTitles(prompter: WizardPrompter): string[] {
  const note = prompter.note as ReturnType<typeof vi.fn>;
  return note.mock.calls
    .map((call) => call[1])
    .filter((title): title is string => typeof title === "string");
}

describe("runOnboardingWizard", () => {
  let suiteRoot = "";
  let suiteCase = 0;

  beforeAll(async () => {
    suiteRoot = await fs.mkdtemp(path.join(os.tmpdir(), "marv-onboard-suite-"));
  });

  afterAll(async () => {
    await fs.rm(suiteRoot, { recursive: true, force: true });
    suiteRoot = "";
    suiteCase = 0;
  });

  beforeEach(() => {
    promptAuthChoiceGrouped.mockReset();
    promptAuthChoiceGrouped.mockResolvedValue("skip");
    applyAuthChoice.mockReset();
    applyAuthChoice.mockImplementation(async (args) => ({ config: args.config }));
    applyPrimaryModel.mockReset();
    applyPrimaryModel.mockImplementation((cfg) => cfg);
    promptDefaultModel.mockReset();
    promptDefaultModel.mockResolvedValue({ config: null, model: null });
    configureGatewayForOnboarding.mockReset();
    configureGatewayForOnboarding.mockImplementation(async (args) => ({
      nextConfig: args.nextConfig,
      settings: {
        port: args.localPort ?? 18789,
        bind: "loopback",
        authMode: "token",
        gatewayToken: "test-token",
        tailscaleMode: "off",
        tailscaleResetOnExit: false,
      },
    }));
    runAuthProbes.mockReset();
    runAuthProbes.mockResolvedValue({
      startedAt: 0,
      finishedAt: 1,
      durationMs: 1,
      totalTargets: 1,
      options: {
        provider: "openai",
        timeoutMs: 8000,
        concurrency: 1,
        maxTokens: 16,
      },
      results: [
        {
          provider: "openai",
          model: "openai/gpt-5.2",
          label: "default",
          source: "env",
          status: "ok",
        },
      ],
    });
    describeProbeSummary.mockReset();
    describeProbeSummary.mockReturnValue("Probed 1 target in 1ms");
    setupSkills.mockReset();
    setupSkills.mockImplementation(async (cfg) => cfg);
    ensureWorkspaceAndSessions.mockReset();
    ensureWorkspaceAndSessions.mockImplementation(async () => {});
    writeConfigFile.mockReset();
    writeConfigFile.mockImplementation(async () => {});
    readConfigFileSnapshot.mockReset();
    readConfigFileSnapshot.mockResolvedValue({
      path: "/tmp/.marv/marv.json",
      exists: false,
      raw: null,
      parsed: {},
      resolved: {},
      valid: true,
      config: {},
      issues: [],
      warnings: [],
      legacyIssues: [],
    });
    healthCommand.mockReset();
    healthCommand.mockImplementation(async () => {});
    runTui.mockReset();
    runTui.mockImplementation(async () => {});
    finalizeOnboardingWizard.mockClear();
  });

  async function makeCaseDir(prefix: string): Promise<string> {
    const dir = path.join(suiteRoot, `${prefix}${++suiteCase}`);
    await fs.mkdir(dir, { recursive: true });
    return dir;
  }

  it("exits when config is invalid", async () => {
    readConfigFileSnapshot.mockResolvedValueOnce({
      path: "/tmp/.marv/marv.json",
      exists: true,
      raw: "{}",
      parsed: {},
      resolved: {},
      valid: false,
      config: {},
      issues: [{ path: "routing.allowFrom", message: "Legacy key" }],
      warnings: [],
      legacyIssues: [{ path: "routing.allowFrom", message: "Legacy key" }],
    });

    const select = vi.fn(
      async (_params: WizardSelectParams<unknown>) => "quickstart",
    ) as unknown as WizardPrompter["select"];
    const prompter = createWizardPrompter({ select });
    const runtime = createRuntime({ throwsOnExit: true });

    await expect(
      runOnboardingWizard(
        {
          acceptRisk: true,
          flow: "quickstart",
          authChoice: "skip",
          installDaemon: false,
          skipChannels: true,
          skipSkills: true,
          skipHealth: true,
          skipUi: true,
        },
        runtime,
        prompter,
      ),
    ).rejects.toThrow("exit:1");

    expect(select).not.toHaveBeenCalled();
    expect(prompter.outro).toHaveBeenCalled();
  });

  it("skips prompts and setup steps when flags are set", async () => {
    const select = vi.fn(
      async (_params: WizardSelectParams<unknown>) => "quickstart",
    ) as unknown as WizardPrompter["select"];
    const multiselect: WizardPrompter["multiselect"] = vi.fn(async () => []);
    const prompter = createWizardPrompter({ select, multiselect });
    const runtime = createRuntime({ throwsOnExit: true });

    await runOnboardingWizard(
      {
        acceptRisk: true,
        flow: "quickstart",
        authChoice: "skip",
        installDaemon: false,
        skipChannels: true,
        skipSkills: true,
        skipHealth: true,
        skipUi: true,
      },
      runtime,
      prompter,
    );

    expect(select).not.toHaveBeenCalled();
    expect(setupChannels).not.toHaveBeenCalled();
    expect(setupSkills).not.toHaveBeenCalled();
    expect(healthCommand).not.toHaveBeenCalled();
    expect(runTui).not.toHaveBeenCalled();
  });

  async function runTuiHatchTest(params: {
    writeBootstrapFile: boolean;
    expectedMessage: string | undefined;
  }) {
    runTui.mockClear();

    const workspaceDir = await makeCaseDir("workspace-");
    if (params.writeBootstrapFile) {
      await fs.writeFile(path.join(workspaceDir, DEFAULT_BOOTSTRAP_FILENAME), "{}");
    }

    const select = vi.fn(async (opts: WizardSelectParams<unknown>) => {
      if (opts.message === "How do you want to hatch your bot?") {
        return "tui";
      }
      return "quickstart";
    }) as unknown as WizardPrompter["select"];

    const prompter = createWizardPrompter({ select });
    const runtime = createRuntime({ throwsOnExit: true });

    await runOnboardingWizard(
      {
        acceptRisk: true,
        flow: "quickstart",
        mode: "local",
        workspace: workspaceDir,
        authChoice: "skip",
        skipChannels: true,
        skipSkills: true,
        skipHealth: true,
        installDaemon: false,
      },
      runtime,
      prompter,
    );

    expect(runTui).toHaveBeenCalledWith(
      expect.objectContaining({
        deliver: false,
        message: params.expectedMessage,
      }),
    );
  }

  it("launches TUI without auto-delivery when hatching", async () => {
    await runTuiHatchTest({ writeBootstrapFile: true, expectedMessage: "Wake up, my friend!" });
  });

  it("offers TUI hatch even without BOOTSTRAP.md", async () => {
    await runTuiHatchTest({ writeBootstrapFile: false, expectedMessage: undefined });
  });

  it("shows the web search hint at the end of onboarding", async () => {
    const prevBraveKey = process.env.BRAVE_API_KEY;
    delete process.env.BRAVE_API_KEY;

    try {
      const note: WizardPrompter["note"] = vi.fn(async () => {});
      const prompter = createWizardPrompter({ note });
      const runtime = createRuntime();

      await runOnboardingWizard(
        {
          acceptRisk: true,
          flow: "quickstart",
          authChoice: "skip",
          installDaemon: false,
          skipChannels: true,
          skipSkills: true,
          skipHealth: true,
          skipUi: true,
        },
        runtime,
        prompter,
      );

      const calls = (note as unknown as { mock: { calls: unknown[][] } }).mock.calls;
      expect(calls.length).toBeGreaterThan(0);
      expect(calls.some((call) => call?.[1] === "Web search (optional)")).toBe(true);
    } finally {
      if (prevBraveKey === undefined) {
        delete process.env.BRAVE_API_KEY;
      } else {
        process.env.BRAVE_API_KEY = prevBraveKey;
      }
    }
  });

  it("logs wizard metrics when instrumentation is enabled", async () => {
    const prev = process.env.MARV_WIZARD_METRICS;
    process.env.MARV_WIZARD_METRICS = "1";

    try {
      const prompter = createWizardPrompter();
      const runtime = createRuntime();

      await runOnboardingWizard(
        {
          acceptRisk: true,
          flow: "quickstart",
          authChoice: "skip",
          installDaemon: false,
          skipChannels: true,
          skipSkills: true,
          skipHealth: true,
          skipUi: true,
        },
        runtime,
        prompter,
      );

      expect(runtime.log).toHaveBeenCalledWith(expect.stringContaining("Wizard metrics:"));
    } finally {
      if (prev === undefined) {
        delete process.env.MARV_WIZARD_METRICS;
      } else {
        process.env.MARV_WIZARD_METRICS = prev;
      }
    }
  });

  it("keeps the fully skipped quickstart path to the P0 prompt budget", async () => {
    const prev = process.env.MARV_WIZARD_METRICS;
    process.env.MARV_WIZARD_METRICS = "1";

    try {
      const prompter = createWizardPrompter();
      const runtime = createRuntime();

      await runOnboardingWizard(
        {
          acceptRisk: true,
          flow: "quickstart",
          authChoice: "skip",
          installDaemon: false,
          skipChannels: true,
          skipSkills: true,
          skipHealth: true,
          skipUi: true,
        },
        runtime,
        prompter,
      );

      expect(parseWizardMetricsLog(runtime)).toEqual({
        interactions: 3,
        total: 12,
      });
    } finally {
      if (prev === undefined) {
        delete process.env.MARV_WIZARD_METRICS;
      } else {
        process.env.MARV_WIZARD_METRICS = prev;
      }
    }
  });

  it("keeps the TUI hatch path within the quickstart interaction budget", async () => {
    const prev = process.env.MARV_WIZARD_METRICS;
    process.env.MARV_WIZARD_METRICS = "1";

    try {
      runTui.mockClear();
      const workspaceDir = await makeCaseDir("metrics-workspace-");
      await fs.writeFile(path.join(workspaceDir, DEFAULT_BOOTSTRAP_FILENAME), "{}");

      const select = vi.fn(async (opts: WizardSelectParams<unknown>) => {
        if (opts.message === "How do you want to hatch your bot?") {
          return "tui";
        }
        return "quickstart";
      }) as unknown as WizardPrompter["select"];

      const prompter = createWizardPrompter({ select });
      const runtime = createRuntime();

      await runOnboardingWizard(
        {
          acceptRisk: true,
          flow: "quickstart",
          mode: "local",
          workspace: workspaceDir,
          authChoice: "skip",
          skipChannels: true,
          skipSkills: true,
          skipHealth: true,
          installDaemon: false,
        },
        runtime,
        prompter,
      );

      const metrics = parseWizardMetricsLog(runtime);
      expect(metrics.interactions).toBe(4);
      expect(metrics.total).toBeGreaterThanOrEqual(12);
      expect(metrics.total).toBeLessThanOrEqual(13);
    } finally {
      if (prev === undefined) {
        delete process.env.MARV_WIZARD_METRICS;
      } else {
        process.env.MARV_WIZARD_METRICS = prev;
      }
    }
  });

  it("resolves model setup before asking for the local workspace path", async () => {
    const events: string[] = [];
    promptAuthChoiceGrouped.mockImplementationOnce(async () => {
      events.push("auth");
      return "skip";
    });
    promptDefaultModel.mockImplementationOnce(async () => {
      events.push("model");
      return { config: null, model: null };
    });

    const select = vi.fn(async (params: WizardSelectParams<unknown>) => {
      if (params.message === "Onboarding mode") {
        return "advanced";
      }
      if (params.message === "What do you want to set up?") {
        return "local";
      }
      return "quickstart";
    }) as unknown as WizardPrompter["select"];
    const text: WizardPrompter["text"] = vi.fn(async (params) => {
      if (params.message === "Workspace directory") {
        events.push("workspace");
      }
      return "/tmp/model-first-workspace";
    });

    const prompter = createWizardPrompter({ select, text });
    const runtime = createRuntime();

    await runOnboardingWizard(
      {
        acceptRisk: true,
        skipChannels: true,
        skipSkills: true,
        skipHealth: true,
        skipUi: true,
      },
      runtime,
      prompter,
    );

    expect(events).toEqual(["auth", "model", "workspace"]);
  });

  it("validates a configured model before prompting for the local workspace path", async () => {
    const events: string[] = [];
    promptAuthChoiceGrouped.mockImplementationOnce(async () => {
      events.push("auth");
      return "token";
    });
    promptDefaultModel.mockImplementationOnce(async () => {
      events.push("model");
      return { config: null, model: "openai/gpt-5.2" };
    });
    applyPrimaryModel.mockImplementationOnce((cfg, model) => {
      events.push("apply-model");
      return {
        ...cfg,
        agents: {
          ...cfg?.agents,
          defaults: {
            ...cfg?.agents?.defaults,
            model: { primary: model },
          },
        },
      };
    });
    runAuthProbes.mockImplementationOnce(async () => {
      events.push("probe");
      return {
        startedAt: 0,
        finishedAt: 1,
        durationMs: 1,
        totalTargets: 1,
        options: {
          provider: "openai",
          timeoutMs: 8000,
          concurrency: 1,
          maxTokens: 16,
        },
        results: [
          {
            provider: "openai",
            model: "openai/gpt-5.2",
            label: "default",
            source: "env",
            status: "ok",
          },
        ],
      };
    });

    const select = vi.fn(async (params: WizardSelectParams<unknown>) => {
      if (params.message === "Onboarding mode") {
        return "advanced";
      }
      if (params.message === "What do you want to set up?") {
        return "local";
      }
      return "quickstart";
    }) as unknown as WizardPrompter["select"];
    const text: WizardPrompter["text"] = vi.fn(async (params) => {
      if (params.message === "Workspace directory") {
        events.push("workspace");
      }
      return "/tmp/model-validated-workspace";
    });

    const prompter = createWizardPrompter({ select, text });
    const runtime = createRuntime();

    await runOnboardingWizard(
      {
        acceptRisk: true,
        skipChannels: true,
        skipSkills: true,
        skipHealth: true,
        skipUi: true,
      },
      runtime,
      prompter,
    );

    expect(events).toEqual(["auth", "model", "apply-model", "probe", "workspace"]);
    expect(runAuthProbes).toHaveBeenCalledWith(
      expect.objectContaining({
        providers: ["openai"],
        modelCandidates: ["openai/gpt-5.2"],
        options: expect.objectContaining({
          provider: "openai",
          timeoutMs: 8000,
          concurrency: 1,
          maxTokens: 16,
        }),
      }),
    );
  });

  it("announces the staged local flow in order", async () => {
    const prompter = createWizardPrompter();
    const runtime = createRuntime();

    await runOnboardingWizard(
      {
        acceptRisk: true,
        flow: "quickstart",
        authChoice: "skip",
        installDaemon: false,
        skipChannels: true,
        skipSkills: true,
        skipHealth: true,
        skipUi: true,
      },
      runtime,
      prompter,
    );

    expect(getWizardNoteTitles(prompter)).toEqual(
      expect.arrayContaining([
        "Stage 1 - Environment and risk",
        "Stage 2 - Model-first activation",
        "Stage 3 - Structured setup",
        "Stage 4 - Review and activation",
      ]),
    );
  });

  it("keeps the advanced reconfiguration path within the measured interaction budget", async () => {
    const prev = process.env.MARV_WIZARD_METRICS;
    process.env.MARV_WIZARD_METRICS = "1";

    try {
      promptAuthChoiceGrouped.mockResolvedValueOnce("token");
      promptDefaultModel.mockResolvedValueOnce({ config: null, model: "openai/gpt-5.2" });
      applyPrimaryModel.mockImplementationOnce((cfg, model) => ({
        ...cfg,
        agents: {
          ...cfg?.agents,
          defaults: {
            ...cfg?.agents?.defaults,
            model: { primary: model },
          },
        },
      }));
      readConfigFileSnapshot.mockResolvedValueOnce({
        path: "/tmp/.marv/marv.json",
        exists: true,
        raw: "{}",
        parsed: {},
        resolved: {},
        valid: true,
        config: {
          agents: { defaults: { workspace: "/tmp/existing-workspace" } },
        },
        issues: [],
        warnings: [],
        legacyIssues: [],
      });

      const select = vi.fn(async (params: WizardSelectParams<unknown>) => {
        if (params.message === "Onboarding mode") {
          return "advanced";
        }
        if (params.message === "Config handling") {
          return "modify";
        }
        if (params.message === "What do you want to set up?") {
          return "local";
        }
        return "quickstart";
      }) as unknown as WizardPrompter["select"];
      const text: WizardPrompter["text"] = vi.fn(async () => "/tmp/existing-workspace");

      const prompter = createWizardPrompter({ select, text });
      const runtime = createRuntime();

      await runOnboardingWizard(
        {
          acceptRisk: true,
          skipChannels: true,
          skipSkills: true,
          skipHealth: true,
          skipUi: true,
        },
        runtime,
        prompter,
      );

      const metrics = parseWizardMetricsLog(runtime);
      expect(metrics.interactions).toBeLessThanOrEqual(11);
      expect(metrics.total).toBeGreaterThanOrEqual(metrics.interactions);
    } finally {
      if (prev === undefined) {
        delete process.env.MARV_WIZARD_METRICS;
      } else {
        process.env.MARV_WIZARD_METRICS = prev;
      }
    }
  });

  it("reuses existing local setup without re-prompting for model or workspace details", async () => {
    readConfigFileSnapshot.mockResolvedValueOnce({
      path: "/tmp/.marv/marv.json",
      exists: true,
      raw: "{}",
      parsed: {},
      resolved: {},
      valid: true,
      config: {
        agents: {
          defaults: {
            workspace: "/tmp/existing-workspace",
            model: { primary: "openai/gpt-5.2" },
          },
        },
        gateway: {
          port: 18789,
          bind: "loopback",
          auth: { mode: "token", token: "existing-token" },
        },
      },
      issues: [],
      warnings: [],
      legacyIssues: [],
    });

    const select = vi.fn(async (params: WizardSelectParams<unknown>) => {
      if (params.message === "Onboarding mode") {
        return "advanced";
      }
      if (params.message === "Config handling") {
        return "keep";
      }
      if (params.message === "What do you want to set up?") {
        throw new Error("setup target prompt should be skipped");
      }
      return "quickstart";
    }) as unknown as WizardPrompter["select"];
    const text: WizardPrompter["text"] = vi.fn(async () => {
      throw new Error("workspace prompt should be skipped");
    });

    const prompter = createWizardPrompter({ select, text });
    const runtime = createRuntime();

    await runOnboardingWizard(
      {
        acceptRisk: true,
        skipChannels: true,
        skipSkills: true,
        skipHealth: true,
        skipUi: true,
      },
      runtime,
      prompter,
    );

    expect(promptAuthChoiceGrouped).not.toHaveBeenCalled();
    expect(promptDefaultModel).not.toHaveBeenCalled();
    expect(configureGatewayForOnboarding).not.toHaveBeenCalled();
    expect(setupSkills).not.toHaveBeenCalled();
    expect(runAuthProbes).toHaveBeenCalledOnce();
  });
});

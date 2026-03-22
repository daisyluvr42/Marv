import { resolveDefaultAgentId } from "../agents/agent-scope.js";
import { DEFAULT_PROVIDER } from "../agents/defaults.js";
import { parseModelRef } from "../agents/model/model-selection.js";
import { buildSoulContent, writeSoulFile } from "../agents/soul.js";
import { formatCliCommand } from "../cli/command-format.js";
import { resolvePrimaryModel } from "../commands/model-default.js";
import { describeProbeSummary, runAuthProbes } from "../commands/models/list.probe.js";
import type {
  GatewayAuthChoice,
  OnboardMode,
  OnboardOptions,
  ResetScope,
} from "../commands/onboard-types.js";
import type { MarvConfig } from "../core/config/config.js";
import {
  DEFAULT_GATEWAY_PORT,
  readConfigFileSnapshot,
  resolveGatewayPort,
  writeConfigFile,
} from "../core/config/config.js";
import type { RuntimeEnv } from "../runtime.js";
import { defaultRuntime } from "../runtime.js";
import { resolveUserPath } from "../utils.js";
import { formatWizardMetrics, instrumentWizardPrompter } from "./metrics.js";
import { presentWizardStage } from "./onboarding.stages.js";
import type {
  GatewayWizardSettings,
  QuickstartGatewayDefaults,
  WizardFlow,
} from "./onboarding.types.js";
import {
  WizardBackSignal,
  WizardCancelledError,
  withBackSupport,
  type WizardPrompter,
} from "./prompts.js";
import type { WizardStepDef } from "./step-runner.js";
import { runStepsWithBack } from "./step-runner.js";

// ---------------------------------------------------------------------------
// Shared helpers (unchanged)
// ---------------------------------------------------------------------------

async function requireRiskAcknowledgement(params: {
  opts: OnboardOptions;
  prompter: WizardPrompter;
}) {
  if (params.opts.acceptRisk === true) {
    return;
  }

  await params.prompter.note(
    [
      "Security warning — please read.",
      "",
      "Marv is a hobby project and still in beta. Expect sharp edges.",
      "This bot can read files and run actions if tools are enabled.",
      "A bad prompt can trick it into doing unsafe things.",
      "",
      "If you're not comfortable with basic security and access control, don't run Marv.",
      "Ask someone experienced to help before enabling tools or exposing it to the internet.",
      "",
      "Recommended baseline:",
      "- Pairing/allowlists + mention gating.",
      "- Sandbox + least-privilege tools.",
      "- Keep secrets out of the agent's reachable filesystem.",
      "- Use the strongest available model for any bot with tools or untrusted inboxes.",
      "",
      "Run regularly:",
      "marv security audit --deep",
      "marv security audit --fix",
      "",
      "Must read: /gateway/security",
    ].join("\n"),
    "Security",
  );

  const ok = await params.prompter.confirm({
    message: "I understand this is powerful and inherently risky. Continue?",
    initialValue: false,
  });
  if (!ok) {
    throw new WizardCancelledError("risk not accepted");
  }
}

async function validateConfiguredModelEarly(params: {
  config: MarvConfig;
  prompter: WizardPrompter;
}) {
  const configured = resolvePrimaryModel(params.config.agents?.defaults?.model)?.trim();
  if (!configured) {
    return;
  }

  const parsed = parseModelRef(configured, DEFAULT_PROVIDER);
  if (!parsed) {
    return;
  }

  const progress = params.prompter.progress("Model check");
  try {
    progress.update(`Checking ${parsed.provider}/${parsed.model}…`);
    const summary = await runAuthProbes({
      cfg: params.config,
      providers: [parsed.provider],
      modelCandidates: [`${parsed.provider}/${parsed.model}`],
      options: {
        provider: parsed.provider,
        timeoutMs: 8_000,
        concurrency: 1,
        maxTokens: 16,
      },
    });
    const ok = summary.results.find((result) => result.status === "ok");
    if (ok) {
      await params.prompter.note(
        [
          `Validated model: ${ok.model ?? `${parsed.provider}/${parsed.model}`}`,
          describeProbeSummary(summary),
        ].join("\n"),
        "Model ready",
      );
      return;
    }

    const firstFailure = summary.results[0];
    await params.prompter.note(
      [
        `Could not validate ${parsed.provider}/${parsed.model} yet.`,
        describeProbeSummary(summary),
        firstFailure?.error ? `Reason: ${firstFailure.error}` : undefined,
        `Check later: ${formatCliCommand("marv models list --probe")}`,
      ]
        .filter(Boolean)
        .join("\n"),
      "Model check",
    );
  } catch (error) {
    await params.prompter.note(
      [
        `Could not validate ${parsed.provider}/${parsed.model} yet.`,
        error instanceof Error ? error.message : String(error),
        `Check later: ${formatCliCommand("marv models list --probe")}`,
      ].join("\n"),
      "Model check",
    );
  } finally {
    progress.stop("Model check complete.");
  }
}

function inferExistingOnboardMode(config: MarvConfig): OnboardMode | undefined {
  const hasRemote = Boolean(config.gateway?.remote?.url?.trim());
  const hasLocalWorkspace = Boolean(config.agents?.defaults?.workspace?.trim());
  const hasLocalGateway =
    typeof config.gateway?.port === "number" ||
    config.gateway?.bind !== undefined ||
    config.gateway?.auth?.mode !== undefined ||
    Boolean(config.gateway?.auth?.token) ||
    Boolean(config.gateway?.auth?.password) ||
    config.gateway?.customBindHost !== undefined ||
    config.gateway?.tailscale?.mode !== undefined;
  const hasLocal = hasLocalWorkspace || hasLocalGateway;
  if (hasLocal && !hasRemote) {
    return "local";
  }
  if (hasRemote && !hasLocal) {
    return "remote";
  }
  return undefined;
}

// Inline soul baseline for seeding during onboarding.
// Mirrors the essential behavioral guidance from docs/reference/templates/SOUL.md
// so new users get a functional agent persona even before workspace bootstrap runs.
const DEFAULT_P0_SOUL_BASELINE = [
  "Be genuinely helpful, not performatively helpful. Skip filler words — just help.",
  "Have opinions. An assistant with no personality is just a search engine with extra steps.",
  "Be resourceful before asking. Try to figure it out — read files, check context, search. Then ask if stuck.",
  "Earn trust through competence. Be careful with external actions; be bold with internal ones.",
  "Remember you're a guest in someone's life. Treat it with respect.",
  'Before saying a task cannot be done, attempt concrete exploration steps. Only conclude "cannot" with a specific blocker.',
  "Be the assistant you'd actually want to talk to. Concise when needed, thorough when it matters.",
].join("\n");

/**
 * Build a default Soul personalized with the agent name chosen during onboarding.
 * Uses the inline baseline so it works regardless of workspace/template state.
 */
export function buildDefaultP0Soul(agentName: string): string {
  const name = agentName.trim();
  if (!name) {
    return DEFAULT_P0_SOUL_BASELINE;
  }
  return `Your name is ${name}.\n\n${DEFAULT_P0_SOUL_BASELINE}`;
}

async function promptAgentP0ForOnboarding(params: {
  config: MarvConfig;
  opts: OnboardOptions;
  prompter: WizardPrompter;
}): Promise<MarvConfig> {
  const identity = await params.prompter.text({
    message: "Agent name",
    placeholder: "What should the agent be called?",
    initialValue: params.opts.p0Identity ?? "",
  });
  const user = await params.prompter.text({
    message: "How should the agent address you?",
    placeholder: "Your name or preferred nickname",
    initialValue: params.opts.p0User ?? "",
  });
  // Seed Soul from the baseline template when no explicit soul is set,
  // personalized with the agent's chosen name.
  let soul = params.opts.p0Soul;
  if (!soul?.trim()) {
    soul = buildDefaultP0Soul(String(identity ?? ""));
  }
  const agentId = resolveDefaultAgentId(params.config);
  const content = buildSoulContent({
    soul,
    identity: String(identity ?? ""),
    user: String(user ?? ""),
  });
  await writeSoulFile(agentId, content);
  return params.config;
}

// ---------------------------------------------------------------------------
// Step state & context types
// ---------------------------------------------------------------------------

function buildQuickstartGateway(baseConfig: MarvConfig): QuickstartGatewayDefaults {
  const hasExisting =
    typeof baseConfig.gateway?.port === "number" ||
    baseConfig.gateway?.bind !== undefined ||
    baseConfig.gateway?.auth?.mode !== undefined ||
    baseConfig.gateway?.auth?.token !== undefined ||
    baseConfig.gateway?.auth?.password !== undefined ||
    baseConfig.gateway?.customBindHost !== undefined ||
    baseConfig.gateway?.tailscale?.mode !== undefined;

  const bindRaw = baseConfig.gateway?.bind;
  const bind =
    bindRaw === "loopback" ||
    bindRaw === "lan" ||
    bindRaw === "auto" ||
    bindRaw === "custom" ||
    bindRaw === "tailnet"
      ? bindRaw
      : "loopback";

  let authMode: GatewayAuthChoice = "token";
  if (baseConfig.gateway?.auth?.mode === "token" || baseConfig.gateway?.auth?.mode === "password") {
    authMode = baseConfig.gateway.auth.mode;
  } else if (baseConfig.gateway?.auth?.token) {
    authMode = "token";
  } else if (baseConfig.gateway?.auth?.password) {
    authMode = "password";
  }

  const tailscaleRaw = baseConfig.gateway?.tailscale?.mode;
  const tailscaleMode =
    tailscaleRaw === "off" || tailscaleRaw === "serve" || tailscaleRaw === "funnel"
      ? tailscaleRaw
      : "off";

  return {
    hasExisting,
    port: resolveGatewayPort(baseConfig),
    bind,
    authMode,
    tailscaleMode,
    token: baseConfig.gateway?.auth?.token,
    password: baseConfig.gateway?.auth?.password,
    customBindHost: baseConfig.gateway?.customBindHost,
    tailscaleResetOnExit: baseConfig.gateway?.tailscale?.resetOnExit ?? false,
  };
}

/** JSON-serializable state passed through the step runner. */
type OnboardStepState = {
  flow: WizardFlow | undefined;
  mode: OnboardMode | undefined;
  reuseExistingLocalSetup: boolean;
  baseConfig: MarvConfig;
  nextConfig: MarvConfig;
  workspaceDir: string;
  quickstartGateway: QuickstartGatewayDefaults;
  settings: GatewayWizardSettings | undefined;
  snapshotExists: boolean;
  launchedTui: boolean;
};

/** Non-serializable context shared across steps. */
type OnboardStepCtx = {
  opts: OnboardOptions;
  runtime: RuntimeEnv;
  /** Back-wrapped prompter (adds "← Back" to select prompts). */
  prompter: WizardPrompter;
  /** Original prompter without back wrapper (for preamble / non-backable prompts). */
  rawPrompter: WizardPrompter;
  onboardHelpers: typeof import("../commands/onboard-helpers.js");
};

// ---------------------------------------------------------------------------
// Step functions
// ---------------------------------------------------------------------------

/** Step 1: Flow selection (quickstart / manual). */
async function stepFlowSelection(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  const quickstartHint = `Configure details later via ${formatCliCommand("marv configure")}.`;
  const manualHint = "Configure port, network, Tailscale, and auth options.";
  let flow: WizardFlow =
    ctx.opts.flow ??
    (await ctx.prompter.select({
      message: "Onboarding mode",
      options: [
        { value: "quickstart", label: "QuickStart", hint: quickstartHint },
        { value: "advanced", label: "Manual", hint: manualHint },
      ],
      initialValue: "quickstart",
    }));

  if (ctx.opts.mode === "remote" && flow === "quickstart") {
    await ctx.rawPrompter.note(
      "QuickStart only supports local gateways. Switching to Manual mode.",
      "QuickStart",
    );
    flow = "advanced";
  }

  return { ...state, flow };
}

/** Step 2: Existing config handling (keep / modify / reset). */
async function stepConfigHandling(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  await ctx.rawPrompter.note(
    ctx.onboardHelpers.summarizeExistingConfig(state.baseConfig),
    "Existing config detected",
  );

  const action = await ctx.prompter.select({
    message: "Config handling",
    options: [
      { value: "keep", label: "Use existing values" },
      { value: "modify", label: "Update values" },
      { value: "reset", label: "Reset" },
    ],
  });

  let baseConfig = state.baseConfig;
  let reuseExistingLocalSetup = false;

  if (action === "reset") {
    const workspaceDefault =
      baseConfig.agents?.defaults?.workspace ?? ctx.onboardHelpers.DEFAULT_WORKSPACE;
    const resetScope = (await ctx.prompter.select({
      message: "Reset scope",
      options: [
        { value: "config", label: "Config only" },
        {
          value: "config+creds+sessions",
          label: "Config + creds + sessions",
        },
        {
          value: "full",
          label: "Full reset (config + creds + sessions + workspace)",
        },
      ],
    })) as ResetScope;
    await ctx.onboardHelpers.handleReset(
      resetScope,
      resolveUserPath(workspaceDefault),
      ctx.runtime,
    );
    baseConfig = {};
  } else if (action === "keep") {
    reuseExistingLocalSetup = true;
  }

  const quickstartGateway = buildQuickstartGateway(baseConfig);
  return {
    ...state,
    baseConfig,
    nextConfig: baseConfig,
    reuseExistingLocalSetup,
    quickstartGateway,
  };
}

/** Step 3: Mode selection (local / remote) + quickstart summary note. */
async function stepModeSelection(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  const { flow, baseConfig, reuseExistingLocalSetup, quickstartGateway } = state;

  // Show quickstart summary if applicable.
  if (flow === "quickstart") {
    const formatBind = (value: "loopback" | "lan" | "auto" | "custom" | "tailnet") => {
      if (value === "loopback") {
        return "Loopback (127.0.0.1)";
      }
      if (value === "lan") {
        return "LAN";
      }
      if (value === "custom") {
        return "Custom IP";
      }
      if (value === "tailnet") {
        return "Tailnet (Tailscale IP)";
      }
      return "Auto";
    };
    const formatAuth = (value: GatewayAuthChoice) =>
      value === "token" ? "Token (default)" : "Password";
    const formatTailscale = (value: "off" | "serve" | "funnel") => {
      if (value === "off") {
        return "Off";
      }
      if (value === "serve") {
        return "Serve";
      }
      return "Funnel";
    };
    const quickstartLines = quickstartGateway.hasExisting
      ? [
          "Keeping your current gateway settings:",
          `Gateway port: ${quickstartGateway.port}`,
          `Gateway bind: ${formatBind(quickstartGateway.bind)}`,
          ...(quickstartGateway.bind === "custom" && quickstartGateway.customBindHost
            ? [`Gateway custom IP: ${quickstartGateway.customBindHost}`]
            : []),
          `Gateway auth: ${formatAuth(quickstartGateway.authMode)}`,
          `Tailscale exposure: ${formatTailscale(quickstartGateway.tailscaleMode)}`,
          "Direct to chat channels.",
        ]
      : [
          `Gateway port: ${DEFAULT_GATEWAY_PORT}`,
          "Gateway bind: Loopback (127.0.0.1)",
          "Gateway auth: Token (default)",
          "Tailscale exposure: Off",
          "Direct to chat channels.",
        ];
    await ctx.rawPrompter.note(quickstartLines.join("\n"), "QuickStart");
  }

  // Probe gateways for mode hints.
  const localPort = resolveGatewayPort(baseConfig);
  const localUrl = `ws://127.0.0.1:${localPort}`;
  const localProbe = await ctx.onboardHelpers.probeGatewayReachable({
    url: localUrl,
    token:
      baseConfig.gateway?.auth?.token ??
      process.env.MARV_GATEWAY_TOKEN ??
      process.env.MARV_GATEWAY_TOKEN,
    password:
      baseConfig.gateway?.auth?.password ??
      process.env.MARV_GATEWAY_PASSWORD ??
      process.env.MARV_GATEWAY_PASSWORD,
  });
  const remoteUrl = baseConfig.gateway?.remote?.url?.trim() ?? "";
  const remoteProbe = remoteUrl
    ? await ctx.onboardHelpers.probeGatewayReachable({
        url: remoteUrl,
        token: baseConfig.gateway?.remote?.token,
      })
    : null;

  const inferredMode =
    reuseExistingLocalSetup && ctx.opts.mode === undefined
      ? inferExistingOnboardMode(baseConfig)
      : undefined;
  if (reuseExistingLocalSetup && inferredMode) {
    await ctx.rawPrompter.note(
      inferredMode === "local"
        ? "Your current setup already looks local, so onboarding will keep using this machine."
        : "Your current setup already looks remote, so onboarding will keep using that gateway.",
      "Setup target",
    );
  }

  const mode =
    ctx.opts.mode ??
    inferredMode ??
    (flow === "quickstart"
      ? "local"
      : ((await ctx.prompter.select({
          message: "What do you want to set up?",
          options: [
            {
              value: "local",
              label: "Local gateway (this machine)",
              hint: localProbe.ok
                ? `Gateway reachable (${localUrl})`
                : `No gateway detected (${localUrl})`,
            },
            {
              value: "remote",
              label: "Remote gateway (info-only)",
              hint: !remoteUrl
                ? "No remote URL configured yet"
                : remoteProbe?.ok
                  ? `Gateway reachable (${remoteUrl})`
                  : `Configured but unreachable (${remoteUrl})`,
            },
          ],
        })) as OnboardMode));

  return { ...state, mode };
}

/** Step 4 (remote only): Remote gateway configuration. */
async function stepRemoteConfig(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  await presentWizardStage(ctx.rawPrompter, "setup", {
    flow: state.flow ?? undefined,
    mode: state.mode ?? undefined,
  });
  const { promptRemoteGatewayConfig } = await import("../commands/onboard-remote.js");
  const nextConfig = await promptRemoteGatewayConfig(state.baseConfig, ctx.prompter);
  return { ...state, nextConfig };
}

/** Step 4 (local, reuse): Validate existing model. */
async function stepReuseModelValidation(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  await presentWizardStage(ctx.rawPrompter, "model", {
    flow: state.flow ?? undefined,
    mode: state.mode ?? undefined,
    reuseExistingLocalSetup: true,
  });
  await ctx.rawPrompter.note(
    "Reusing your existing local model, workspace, gateway, and channel settings.",
    "Using existing setup",
  );
  const settings: GatewayWizardSettings = {
    port: state.quickstartGateway.port,
    bind: state.quickstartGateway.bind,
    customBindHost: state.quickstartGateway.customBindHost,
    authMode: state.quickstartGateway.authMode,
    gatewayToken: state.quickstartGateway.token,
    tailscaleMode: state.quickstartGateway.tailscaleMode,
    tailscaleResetOnExit: state.quickstartGateway.tailscaleResetOnExit,
  };
  await validateConfiguredModelEarly({ config: state.nextConfig, prompter: ctx.rawPrompter });
  return { ...state, settings };
}

/** Step 4 (local, non-reuse): Auth provider + model selection. */
async function stepAuthAndModel(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  await presentWizardStage(ctx.rawPrompter, "model", {
    flow: state.flow ?? undefined,
    mode: state.mode ?? undefined,
  });

  const { promptAuthChoiceGrouped } = await import("../commands/auth-choice-prompt.js");
  const { promptCustomApiConfig } = await import("../commands/onboard-custom.js");
  const { applyAuthChoice, resolvePreferredProviderForAuthChoice, warnIfModelConfigLooksOff } =
    await import("../commands/auth-choice.js");
  const { applyPrimaryModel, promptDefaultModel } = await import("../commands/model-picker.js");

  let nextConfig = state.nextConfig;
  const authChoiceFromPrompt = ctx.opts.authChoice === undefined;
  const authChoice =
    ctx.opts.authChoice ??
    (await promptAuthChoiceGrouped({
      prompter: ctx.prompter,
      includeSkip: true,
    }));

  if (authChoice === "custom-api-key") {
    const customResult = await promptCustomApiConfig({
      prompter: ctx.prompter,
      runtime: ctx.runtime,
      config: nextConfig,
    });
    nextConfig = customResult.config;
  } else {
    const authResult = await applyAuthChoice({
      authChoice,
      config: nextConfig,
      prompter: ctx.prompter,
      runtime: ctx.runtime,
      setDefaultModel: true,
      opts: {
        tokenProvider: ctx.opts.tokenProvider,
        token: ctx.opts.authChoice === "apiKey" && ctx.opts.token ? ctx.opts.token : undefined,
      },
    });
    nextConfig = authResult.config;
  }

  if (authChoiceFromPrompt && authChoice !== "custom-api-key") {
    const modelSelection = await promptDefaultModel({
      config: nextConfig,
      prompter: ctx.prompter,
      allowKeep: true,
      ignoreAllowlist: true,
      includeVllm: true,
      preferredProvider: resolvePreferredProviderForAuthChoice(authChoice),
    });
    if (modelSelection.config) {
      nextConfig = modelSelection.config;
    }
    if (modelSelection.model) {
      nextConfig = applyPrimaryModel(nextConfig, modelSelection.model);
    }
  }

  await warnIfModelConfigLooksOff(nextConfig, ctx.rawPrompter);
  if (authChoice !== "skip") {
    await validateConfiguredModelEarly({ config: nextConfig, prompter: ctx.rawPrompter });
  }

  return { ...state, nextConfig };
}

/** Step 5 (local, non-reuse): Workspace directory + agent P0. */
async function stepWorkspaceAndP0(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  await presentWizardStage(ctx.rawPrompter, "setup", {
    flow: state.flow ?? undefined,
    mode: state.mode ?? undefined,
  });

  const workspaceInput =
    ctx.opts.workspace ??
    (state.flow === "quickstart"
      ? (state.baseConfig.agents?.defaults?.workspace ?? ctx.onboardHelpers.DEFAULT_WORKSPACE)
      : await ctx.prompter.text({
          message: "Workspace directory",
          initialValue:
            state.baseConfig.agents?.defaults?.workspace ?? ctx.onboardHelpers.DEFAULT_WORKSPACE,
        }));

  const workspaceDir = resolveUserPath(
    workspaceInput.trim() || ctx.onboardHelpers.DEFAULT_WORKSPACE,
  );

  const { applyOnboardingLocalWorkspaceConfig } = await import("../commands/onboard-config.js");
  let nextConfig = applyOnboardingLocalWorkspaceConfig(state.nextConfig, workspaceDir);
  nextConfig = await promptAgentP0ForOnboarding({
    config: nextConfig,
    opts: ctx.opts,
    prompter: ctx.prompter,
  });

  return { ...state, nextConfig, workspaceDir };
}

/** Step 6 (local, non-reuse, advanced): Memory search + auto-routing. */
async function stepAdvancedConfig(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  let nextConfig = state.nextConfig;

  const { promptMemorySearchForOnboarding } = await import("../commands/configure.memory.js");
  nextConfig = await promptMemorySearchForOnboarding({
    config: nextConfig,
    prompter: ctx.prompter,
  });

  const { promptAutoRouting } = await import("../commands/onboard-auto-routing.js");
  nextConfig = await promptAutoRouting({ config: nextConfig, prompter: ctx.prompter });

  return { ...state, nextConfig };
}

/** Step 7 (local, non-reuse): Gateway configuration. */
async function stepGateway(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  const localPort = resolveGatewayPort(state.baseConfig);
  const { configureGatewayForOnboarding } = await import("./onboarding.gateway-config.js");
  const gateway = await configureGatewayForOnboarding({
    flow: state.flow!,
    baseConfig: state.baseConfig,
    nextConfig: state.nextConfig,
    localPort,
    quickstartGateway: state.quickstartGateway,
    prompter: ctx.prompter,
    runtime: ctx.runtime,
  });
  return { ...state, nextConfig: gateway.nextConfig, settings: gateway.settings };
}

/** Step 8 (local, non-reuse): Channel setup. */
async function stepChannels(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  if (ctx.opts.skipChannels) {
    await ctx.rawPrompter.note("Skipping channel setup.", "Channels");
    return state;
  }

  const { listChannelPlugins } = await import("../channels/plugins/index.js");
  const { setupChannels } = await import("../commands/onboard-channels.js");
  const quickstartAllowFromChannels =
    state.flow === "quickstart"
      ? listChannelPlugins()
          .filter((plugin) => plugin.meta.quickstartAllowFrom)
          .map((plugin) => plugin.id)
      : [];
  const nextConfig = await setupChannels(state.nextConfig, ctx.runtime, ctx.prompter, {
    allowSignalInstall: true,
    forceAllowFromChannels: quickstartAllowFromChannels,
    skipDmPolicyPrompt: state.flow === "quickstart",
    skipConfirm: state.flow === "quickstart",
    quickstartDefaults: state.flow === "quickstart",
  });
  return { ...state, nextConfig };
}

/** Step 9 (local): Write config + workspace + skills. */
async function stepPersistAndSkills(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  let nextConfig = state.nextConfig;

  if (state.reuseExistingLocalSetup) {
    await presentWizardStage(ctx.rawPrompter, "setup", {
      flow: state.flow ?? undefined,
      mode: state.mode ?? undefined,
      reuseExistingLocalSetup: true,
    });
  }

  await writeConfigFile(nextConfig);
  const { logConfigUpdated } = await import("../core/config/logging.js");
  logConfigUpdated(ctx.runtime);
  await ctx.onboardHelpers.ensureWorkspaceAndSessions(state.workspaceDir, ctx.runtime, {
    skipBootstrap: Boolean(nextConfig.agents?.defaults?.skipBootstrap),
  });

  if (ctx.opts.skipSkills || state.reuseExistingLocalSetup) {
    await ctx.rawPrompter.note(
      state.reuseExistingLocalSetup
        ? "Keeping your existing skills setup."
        : "Skipping skills setup.",
      "Skills",
    );
  } else {
    const { setupSkills } = await import("../commands/onboard-skills.js");
    nextConfig = await setupSkills(nextConfig, state.workspaceDir, ctx.runtime, ctx.prompter);
  }

  return { ...state, nextConfig };
}

/** Final step: Review + finalize. Not backable (no user-facing prompts to redo). */
async function stepFinalize(
  state: OnboardStepState,
  ctx: OnboardStepCtx,
): Promise<OnboardStepState> {
  const mode = state.mode!;

  // Remote path: just persist and finish.
  if (mode === "remote") {
    const nextConfig = ctx.onboardHelpers.applyWizardMetadata(state.nextConfig, {
      command: "onboard",
      mode,
    });
    await writeConfigFile(nextConfig);
    const { logConfigUpdated } = await import("../core/config/logging.js");
    logConfigUpdated(ctx.runtime);
    await ctx.rawPrompter.outro("Remote gateway configured.");
    return { ...state, nextConfig };
  }

  // Local path: persist metadata, then review.
  const nextConfig = ctx.onboardHelpers.applyWizardMetadata(state.nextConfig, {
    command: "onboard",
    mode,
  });
  await writeConfigFile(nextConfig);

  await presentWizardStage(ctx.rawPrompter, "review", {
    flow: state.flow ?? undefined,
    mode,
  });
  const { finalizeOnboardingWizard } = await import("./onboarding.finalize.js");
  const { launchedTui } = await finalizeOnboardingWizard({
    flow: state.flow!,
    opts: ctx.opts,
    baseConfig: state.baseConfig,
    nextConfig,
    workspaceDir: state.workspaceDir,
    settings: state.settings!,
    prompter: ctx.rawPrompter,
    runtime: ctx.runtime,
  });
  return { ...state, nextConfig, launchedTui };
}

// ---------------------------------------------------------------------------
// Main wizard orchestrator
// ---------------------------------------------------------------------------

export async function runOnboardingWizard(
  opts: OnboardOptions,
  runtime: RuntimeEnv = defaultRuntime,
  prompter: WizardPrompter,
) {
  const instrumented = instrumentWizardPrompter(prompter);
  const rawPrompter = instrumented.prompter;
  const backPrompter = withBackSupport(rawPrompter);
  const onboardHelpers = await import("../commands/onboard-helpers.js");

  try {
    // --- Preamble (not backable) ---
    onboardHelpers.printWizardHeader(runtime);
    await rawPrompter.intro("Marv onboarding");
    await presentWizardStage(rawPrompter, "environment", { flow: opts.flow });
    await requireRiskAcknowledgement({ opts, prompter: rawPrompter });

    const snapshot = await readConfigFileSnapshot();
    const baseConfig: MarvConfig = snapshot.valid ? snapshot.config : {};

    if (snapshot.exists && !snapshot.valid) {
      await rawPrompter.note(onboardHelpers.summarizeExistingConfig(baseConfig), "Invalid config");
      if (snapshot.issues.length > 0) {
        await rawPrompter.note(
          [
            ...snapshot.issues.map((iss) => `- ${iss.path}: ${iss.message}`),
            "",
            "Docs: /gateway/configuration",
          ].join("\n"),
          "Config issues",
        );
      }
      await rawPrompter.outro(
        `Config invalid. Run \`${formatCliCommand("marv doctor")}\` to repair it, then re-run onboarding.`,
      );
      runtime.exit(1);
      return;
    }

    // --- Build initial state ---
    const quickstartGateway = buildQuickstartGateway(baseConfig);
    const initialState: OnboardStepState = {
      flow: opts.flow,
      mode: opts.mode,
      reuseExistingLocalSetup: false,
      baseConfig,
      nextConfig: baseConfig,
      workspaceDir: resolveUserPath(
        baseConfig.agents?.defaults?.workspace ?? onboardHelpers.DEFAULT_WORKSPACE,
      ),
      quickstartGateway,
      settings: undefined,
      snapshotExists: snapshot.exists,
      launchedTui: false,
    };

    const ctx: OnboardStepCtx = {
      opts,
      runtime,
      prompter: backPrompter,
      rawPrompter,
      onboardHelpers,
    };

    // --- Build step list ---
    // Steps with guards for conditional execution.
    const isLocal = (s: OnboardStepState) => s.mode === "local";
    const isRemote = (s: OnboardStepState) => s.mode === "remote";
    const isLocalNonReuse = (s: OnboardStepState) =>
      s.mode === "local" && !s.reuseExistingLocalSetup;
    const isLocalReuse = (s: OnboardStepState) => s.mode === "local" && s.reuseExistingLocalSetup;
    const isAdvancedNonReuse = (s: OnboardStepState) =>
      s.mode === "local" && !s.reuseExistingLocalSetup && s.flow !== "quickstart";

    const steps: WizardStepDef<OnboardStepState, OnboardStepCtx>[] = [
      { name: "flow-selection", run: stepFlowSelection },
      { name: "config-handling", run: stepConfigHandling, shouldRun: (s) => s.snapshotExists },
      { name: "mode-selection", run: stepModeSelection },
      // Remote path
      { name: "remote-config", run: stepRemoteConfig, shouldRun: isRemote },
      // Local reuse path
      { name: "reuse-model-validation", run: stepReuseModelValidation, shouldRun: isLocalReuse },
      // Local non-reuse path
      { name: "auth-model", run: stepAuthAndModel, shouldRun: isLocalNonReuse },
      { name: "workspace-p0", run: stepWorkspaceAndP0, shouldRun: isLocalNonReuse },
      { name: "advanced-config", run: stepAdvancedConfig, shouldRun: isAdvancedNonReuse },
      { name: "gateway", run: stepGateway, shouldRun: isLocalNonReuse },
      { name: "channels", run: stepChannels, shouldRun: isLocalNonReuse },
      // Local common
      { name: "persist-skills", run: stepPersistAndSkills, shouldRun: isLocal },
      // Finalize (all paths)
      { name: "finalize", run: stepFinalize },
    ];

    const finalState = await runStepsWithBack(steps, initialState, ctx);

    if (finalState.launchedTui) {
      return;
    }
  } catch (err) {
    // Treat unhandled back signal (at step 0) as cancellation.
    if (err instanceof WizardBackSignal) {
      throw new WizardCancelledError("backed out of wizard");
    }
    throw err;
  } finally {
    if (process.env.MARV_WIZARD_METRICS === "1") {
      runtime.log(formatWizardMetrics(instrumented.getMetrics()));
    }
  }
}
